from zhanlu.backend.analytical_model.op_costmodel.element_op import (
    Element5Prediction,
)
from zhanlu.backend.analytical_model.op_costmodel.matmul import MatmulPrediction
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager

from zhanlu.backend.analytical_model.utils.helper import (
    broadcast_shapes,
    create_op_info,
    merge_flops_dicts,
)
from math import prod


@op_manager.register("MLA")
class MLAPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            q_mul_wuk [b*decode_length, head_num, kv_lora_rank + qk_rope_head_dim]
            cache_lora_kv [batch, prefill_length + decode_length, kv_lora_rank + qk_rope_head_dim]
        """
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.instance = op.instance
        self.q_mul_wuk_dtype, self.cache_lora_kv_dtype = self.inputs_dtype
        self._get_op_size()
        self.init_vector_instruction()

    def _get_op_size(self):
        q_mul_wuk_shape, cache_lora_kv_shape = self.inputs_shape
        self.batch_times_decode_length, self.head_num, _ = q_mul_wuk_shape
        self.batch_size, self.prefill_decode_length, _ = cache_lora_kv_shape
        self.decode_length = self.batch_times_decode_length // self.batch_size
        self.prefill_length = self.prefill_decode_length - self.decode_length

    def init_vector_instruction(self):
        q_shape, kv_shape = self.inputs_shape
        q_dtype, kv_dtype = self.inputs_dtype

        q_seq_len = 1
        batch, head_num, head_dim = q_shape
        _, k_seq_len, _ = kv_shape
        bc = 512 if k_seq_len > 1024 else 256
        m = batch * head_num * q_seq_len
        n = k_seq_len
        dtype_name = str(self.instance.softmax_dtype).split(".")[-1]
        if dtype_name == 'bfloat16':
            dtype_name = 'float16'
        self.vec_instruction = [
            ["VMULS", dtype_name, m, n],
            ["VCGMAX", dtype_name, m, n],
            ["VMAX", dtype_name, m, n / bc],
            ["VSUB", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VSUB", dtype_name, m, n / bc],
            ["VEXP", dtype_name, m, n / bc],
            ["VCGADD", dtype_name, m, n],
            ["VADD", dtype_name, m, n / bc],
            ["VMUL", dtype_name, m, n * head_dim / bc],
            ["VADD", dtype_name, m, n * head_dim / bc],
            ["VCONV", dtype_name, m, n],
        ]
        if q_seq_len > 2:
            self.vec_instruction.append(["VADD", dtype_name, m, n])

    def _create_flash_attention_predictors(self):
        if hasattr(self.instance, "weight_dtype_fp4"):
            self.instance.weight_dtype = self.instance.weight_dtype_fp4
        # q@k
        q_mul_wuk_transpose_shape = [
            self.decode_length,
            self.batch_size,
            self.head_num,
            self.instance.kv_lora_rank + self.instance.qk_rope_head_dim,
        ]
        cache_lora_kv_transpose = [
            self.batch_size,
            self.instance.kv_lora_rank + self.instance.qk_rope_head_dim,
            self.prefill_decode_length,
        ]

        attn_scores_shape = MatmulPrediction.infer_shape(q_mul_wuk_transpose_shape, cache_lora_kv_transpose)

        q_mul_k_op = create_op_info(
            input_list=[q_mul_wuk_transpose_shape, cache_lora_kv_transpose],
            input_dtype_list=[self.instance.weight_dtype, self.instance.weight_dtype],
            output_list=[attn_scores_shape],
            output_dtype_list=[self.instance.weight_dtype],
        )

        qk_prediction = MatmulPrediction(q_mul_k_op, self.hardware)

        # softmax
        softmax_op = create_op_info(
            input_list=[attn_scores_shape],
            input_dtype_list=[self.instance.softmax_dtype],
            output_list=[attn_scores_shape],  # Softmax does not change shape
            output_dtype_list=[self.instance.softmax_dtype],
        )

        softmax_predictor = Element5Prediction(softmax_op, self.hardware)

        # score @ V
        cache_nope_kv_shape = [
            self.batch_size,
            self.prefill_decode_length,
            self.instance.kv_lora_rank,
        ]
        out_put_shape = MatmulPrediction.infer_shape(attn_scores_shape, cache_nope_kv_shape)
        attn_mul_v_op = create_op_info(
            input_list=[attn_scores_shape, cache_nope_kv_shape],
            input_dtype_list=[self.instance.weight_dtype, self.instance.weight_dtype],
            output_list=[out_put_shape],  # Final output shape
            output_dtype_list=[self.instance.output_dtype],
        )
        attn_v_predictor = MatmulPrediction(attn_mul_v_op, self.hardware)

        return qk_prediction, softmax_predictor, attn_v_predictor

    @property
    def cube_flops_dict(self):

        qk_predictor, _, attn_v_predictor = self._create_flash_attention_predictors()

        # Compute FLOPs for Q@K^T multiplication, adjusted for causal mask and number of heads
        qk_flops = {k: v for k, v in qk_predictor.cube_flops_dict.items()}

        # Compute FLOPs for attention scores@V multiplication, adjusted for causal mask and heads
        attn_v_flops = {k: v for k, v in attn_v_predictor.cube_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([qk_flops, attn_v_flops])

        return total_flops_dict

    @property
    def cube_time_dict(self):
        qk_predictor, _, attn_v_predictor = self._create_flash_attention_predictors()

        # Compute FLOPs for Q@K^T multiplication, adjusted for causal mask and number of heads
        qk = {k: v for k, v in qk_predictor.cube_time_dict.items()}

        # Compute FLOPs for attention scores@V multiplication, adjusted for causal mask and heads
        attn_v = {k: v for k, v in attn_v_predictor.cube_time_dict.items()}

        total_time_dict = merge_flops_dicts([qk, attn_v])

        return total_time_dict

    @property
    def vec_flops_dict(self):
        _, softmax_predictor, _ = self._create_flash_attention_predictors()

        softmax_flops_dict = {k: v for k, v in softmax_predictor.vec_flops_dict.items()}

        return softmax_flops_dict

    # TODO 注释掉的代码因为在不同硬件上策略不同，未来会添加判断逻辑
    # @property
    # def memory_size(self):
    #     qk_predictor, softmax_predictor, attn_v_predictor = self._create_flash_attention_predictors()

    #     m_1 = qk_predictor.memory_size
    #     m_2 = softmax_predictor.memory_size
    #     m_3 = attn_v_predictor.memory_size
    #     return m_1 + m_2 + m_3

    # @property
    # def memory_time(self):
    #     qk_predictor, softmax_predictor, attn_v_predictor = self._create_flash_attention_predictors()

    #     m_1 = qk_predictor.memory_time
    #     m_2 = softmax_predictor.memory_time
    #     m_3 = attn_v_predictor.memory_time
    #     return m_1 + m_2 + m_3

    @property
    def op_time(self):
        overlap_softmax = 0.5
        return max([self.cube_time + self.vec_time * overlap_softmax, self.memory_time]) + self.head_tail_time
