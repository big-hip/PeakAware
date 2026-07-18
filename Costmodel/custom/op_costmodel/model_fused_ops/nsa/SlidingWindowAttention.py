from math import prod
from time import gmtime
from xmlrpc.server import DocXMLRPCRequestHandler

import torch
from zhanlu.backend.analytical_model.op_costmodel.element_op import Element5Prediction
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_costmodel.matmul import MatmulPrediction
from zhanlu.backend.analytical_model.op_manager import op_manager
from zhanlu.backend.perf_result import ZhanluPerfResult
from zhanlu.backend.analytical_model.utils.helper import create_op_info, merge_flops_dicts

# 访存只计算了输入输出
# 算力利用率按照两个matmul之和来计算的
# 考虑掩盖度的时候，softmax时间只算计算时间，默认没有访存时间

@op_manager.register("SlidingWindowAttention")
class SlidingWindowAttentionPrediction(BaseOp):
    def __init__(self, op, hardware):
        '''
        Causal mask is assumed since mask input is not available.
        Inputs of the operation:
            q (torch.Tensor): Query tensor with shape [batch, heads, seqlen, head_dim]
            k (torch.Tensor): Key tensor with shape [batch, heads, seqlen, head_dim]
            v (torch.Tensor): Value tensor with shape [batch, heads, seqlen, v_dim]
            mask_ratio (float): Ratio of the mask applied to the attention scores.
            layout (str): layout of input e.g. 'bshd'
        '''
        super().__init__(op, hardware)
        # self.mask_ratio = op.inputs[3]
        # self.layout = op.inputs[4]
        self.mask_ratio = 1
        self.layout = "bshd"
        self.op_type = 'mix'
        self.w = 32

        dim_map = {char: i for i, char in enumerate(self.layout)}

        self.q_shape, self.k_shape, self.v_shape = self.inputs_shape
        self.q_dtype, self.k_dtype, self.v_dtype = self.inputs_dtype

        self.batch = self.q_shape[dim_map['b']] if 'b' in dim_map.keys() else 1
        self.num_heads = self.q_shape[dim_map['h']]
        self.seq_len = self.q_shape[dim_map['s']]
        self.head_dim = self.q_shape[dim_map['d']]
        self.v_dim = self.v_shape[dim_map['d']]

    def _create_flash_attention_predictors(self):
        """
        Create predictors for the three main operations in FlashAttention:
        1. Q @ K^T (Query-Key matrix multiplication)
        2. Softmax(QK^T) (Attention score normalization)
        3. Attention Scores @ V (Value matrix multiplication)

        Returns:
            qk_predictor: Predictor for Q @ K^T operation
            softmax_predictor: Predictor for Softmax operation on attention scores
            attn_v_predictor: Predictor for Attention Scores @ V operation
        """
        q_dtype, k_dtype, v_dtype = self.q_dtype, self.k_dtype, self.v_dtype

        batch, num_heads, seq_len, head_dim = self.batch, self.num_heads, self.seq_len, self.head_dim
        v_dim = self.v_dim

        # Q @ K^T Matrix Multiplication
        q_one_head_shape = [batch, seq_len, head_dim]
        transpose_k_one_head_shape = [batch, head_dim, min(seq_len, self.w+1)]

        q_mul_k_one_head_op = create_op_info(
            input_list=[q_one_head_shape, transpose_k_one_head_shape],
            input_dtype_list=[q_dtype, k_dtype],
            output_list=[[batch, seq_len, seq_len]],
            output_dtype_list=[q_dtype]
        )
        qk_predictor = MatmulPrediction(q_mul_k_one_head_op, self.hardware)

        # Softmax Operation on Attention Scores
        attn_scores_shape = [batch, seq_len, min(seq_len, self.w + 1)]

        # Construct Softmax operation (assuming SoftmaxPrediction class exists)
        softmax_op = create_op_info(
            input_list=[attn_scores_shape],
            input_dtype_list=[torch.float16],
            output_list=[attn_scores_shape],  # Softmax does not change shape
            output_dtype_list=[torch.float16]
        )

        softmax_predictor = Element5Prediction(softmax_op, self.hardware)

        # Attention Scores @ V Matrix Multiplication
        v_one_head_shape = [batch, min(seq_len, self.w + 1), v_dim]

        attn_mul_v_one_head_op = create_op_info(
            input_list=[attn_scores_shape, v_one_head_shape],
            input_dtype_list=[q_dtype, v_dtype],
            output_list=[[batch, seq_len, v_dim]],  # Final output shape
            output_dtype_list=[v_dtype]
        )
        attn_v_predictor = MatmulPrediction(attn_mul_v_one_head_op, self.hardware)

        return qk_predictor, softmax_predictor, attn_v_predictor

    @property
    def cube_flops_dict(self):

        batch, num_heads, seq_len, head_dim = self.batch, self.num_heads, self.seq_len, self.head_dim
        v_dim = self.v_dim

        # Causal mask. It should be got from inputs

        qk_predictor, _, attn_v_predictor = self._create_flash_attention_predictors()

        # Compute FLOPs for Q@K^T multiplication, adjusted for causal mask and number of heads
        qk_flops = {k: v * num_heads * self.mask_ratio for k, v in qk_predictor.cube_flops_dict.items()}

        # Compute FLOPs for attention scores@V multiplication, adjusted for causal mask and heads
        attn_v_flops = {k: v * num_heads * self.mask_ratio for k, v in attn_v_predictor.cube_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([qk_flops, attn_v_flops])

        return total_flops_dict

    @property
    def vector_flops_dict(self):
        # Causal mask. It should be got from inputs

        _, softmax_predictor, _ = self._create_flash_attention_predictors()
        softmax_flops_dict = {k: v * self.mask_ratio for k, v in softmax_predictor.vec_flops_dict.items()}

        return softmax_flops_dict

    @property
    def op_time(self):
        overlap_softmax = 0.5
        return max([self.cube_time + self.vec_time * overlap_softmax, self.memory_time])

@op_manager.register("FlashAttentionActivationBackward")
class FlashAttentionBackwardPrediction(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
            grad_output (torch.Tensor): Value tensor with shape [batch, heads, seqlen, v_dim]
            q (torch.Tensor)
            k (torch.Tensor)
            v (torch.Tensor)
            layout (str)
        output:
            grad_q (torch.Tensor): [batch, heads, seqlen, head_dim]
            grad_k (torch.Tensor): [batch, heads, seqlen, head_dim]
            grad_v (torch.Tensor): [batch, heads, seqlen, v_dim]
        '''
        super().__init__(op, hardware)
        self.layout = op.inputs[4]
        self.op_type = 'mix'
        self.w = 128

        dim_map = {char: i for i, char in enumerate(self.layout)}

        self.q_shape, self.k_shape, self.v_shape = self.outputs_shape
        self.q_dtype, self.k_dtype, self.v_dtype = self.outputs_dtype

        self.batch = self.q_shape[dim_map['b']] if 'b' in dim_map.keys() else 1
        self.num_heads = self.q_shape[dim_map['h']]
        self.seq_len = self.q_shape[dim_map['s']]
        self.head_dim = self.q_shape[dim_map['d']]
        self.v_dim = self.v_shape[dim_map['d']]

    def _create_flash_attention_backward_predictors(self):
        """
        Create predictors for the three main operations in FlashAttention backward:
        1. Q @ K^T (Query-Key matrix multiplication)
        2. Softmax(QK^T) (Attention score normalization)
        3. Attention Scores @ V (Value matrix multiplication)

        Returns:
            qk_predictor: Predictor for Q @ K^T operation
            softmax_predictor: Predictor for Softmax operation on attention scores
            attn_v_predictor: Predictor for Attention Scores @ V operation
        """
        q_dtype, k_dtype, v_dtype = self.q_dtype, self.k_dtype, self.v_dtype
        output_grad_dtype = self.inputs_dtype[0]
        score_dtype = q_dtype # Assuming score dtype is same as q dtype

        batch, num_heads, seq_len, head_dim = self.batch, self.num_heads, self.seq_len, self.head_dim
        v_dim = self.v_dim

        # Attention Scores @ V Matrix Multiplication Grad
        # dl/ds = dl/dattention_weights @ v^T
        # dl/dv = s^T @ dl/dattention_weights
        output_grad_one_head_shape = [batch, seq_len, v_dim]
        transpose_v_one_head_shape = [batch, v_dim, seq_len]

        score_grad_one_head_op = create_op_info(
            input_list=[output_grad_one_head_shape, transpose_v_one_head_shape],
            input_dtype_list=[output_grad_dtype, v_dtype],
            output_list=[[batch, seq_len, seq_len]],  # Final output shape
            output_dtype_list=[score_dtype]
        )
        score_grad_predictor = MatmulPrediction(score_grad_one_head_op, self.hardware)

        transpose_score_one_head_shape = [batch, seq_len, seq_len]
        v_grad_one_head_op = create_op_info(
            input_list=[transpose_score_one_head_shape, output_grad_one_head_shape],
            input_dtype_list=[score_dtype, output_grad_dtype],
            output_list=[[batch, seq_len, v_dim]],  # Final output shape
            output_dtype_list=[v_dtype]
        )
        v_grad_predictor = MatmulPrediction(v_grad_one_head_op, self.hardware)

        # Softmax Operation on Attention Scores Grad
        attn_scores_shape = [batch, seq_len, seq_len]
        softmax_grad_op = create_op_info(
            input_list=[attn_scores_shape],
            input_dtype_list=[torch.float16],  # Assuming softmax operates on float16
            output_list=[attn_scores_shape],
            output_dtype_list=[torch.float16]
        )
        # TODO
        softmax_predictor = Element5Prediction(softmax_grad_op, self.hardware)

        # Q @ K^T Matrix Multiplication Grad
        # dl/dq = dl/ds * k
        # dl/dk = dl/ds^T * q
        q_one_head_shape = [batch, seq_len, head_dim]
        k_one_head_shape = [batch, seq_len, head_dim]
        transpose_attn_scores_shape = [batch, seq_len, seq_len]
        q_grad_one_head_op = create_op_info(
            input_list=[attn_scores_shape, k_one_head_shape],
            input_dtype_list=[score_dtype, k_dtype],
            output_list=[q_one_head_shape],  # Final output shape
            output_dtype_list=[q_dtype]
        )
        q_grad_predictor = MatmulPrediction(q_grad_one_head_op, self.hardware)

        k_grad_one_head_op = create_op_info(
            input_list=[transpose_attn_scores_shape, q_one_head_shape],
            input_dtype_list=[score_dtype, q_dtype],
            output_list=[k_one_head_shape],  # Final output shape
            output_dtype_list=[k_dtype]
        )
        k_grad_predictor = MatmulPrediction(q_grad_one_head_op, self.hardware)
        return score_grad_predictor, v_grad_predictor, softmax_predictor, q_grad_predictor, k_grad_predictor

    @property
    def cube_flops_dict(self):
        batch, num_heads, seq_len, head_dim = self.batch, self.num_heads, self.seq_len, self.head_dim

        # Causal mask. It should be got from inputs
        mask_ratio = 0.5

        score_grad_predictor, v_grad_predictor, _, q_grad_predictor, k_grad_predictor = self._create_flash_attention_backward_predictors()

        score_grad_flops = {k: v * num_heads * mask_ratio for k, v in score_grad_predictor.cube_flops_dict.items()}
        v_grad_flops = {k: v * num_heads * mask_ratio for k, v in v_grad_predictor.cube_flops_dict.items()}
        q_grad_flops = {k: v * num_heads * mask_ratio for k, v in q_grad_predictor.cube_flops_dict.items()}
        k_grad_flops = {k: v * num_heads * mask_ratio for k, v in k_grad_predictor.cube_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([score_grad_flops, v_grad_flops, q_grad_flops, k_grad_flops])

        return total_flops_dict

    @property
    def vec_flops_dict(self):
        # Causal mask. It should be got from inputs
        mask_ratio = 0.5

        _, _, softmax_grad_predictor, _, _ = self._create_flash_attention_predictors()
        softmax_grad_flops_dict = {k: v * mask_ratio for k, v in softmax_grad_predictor.vec_flops_dict.items()}

        return softmax_grad_flops_dict

    @property
    def op_time(self):
        overlap_softmax = 0.25
        return max([self.cube_time + self.vec_time * overlap_softmax, self.memory_time]) + self.head_tail_time