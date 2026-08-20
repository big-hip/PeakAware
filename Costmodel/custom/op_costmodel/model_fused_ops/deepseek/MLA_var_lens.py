from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.analytical_model.utils.helper import (
    create_op_info,
    merge_flops_dicts,
)
from atencost.backend.analytical_model.op_costmodel.matmul import (
    MatmulPrediction,
    VarLenQKMatmulPrediction,
    VarLenScoreVMatmulPrediction
)
from atencost.backend.analytical_model.op_costmodel.element_op import VarLenSoftmaxPrediction
import custom.op_costmodel


@op_manager.register("MLA_var_lens")
class MLAVarLensPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            input0: q_mul_wuk [b*decode_length, head_num, kv_lora_rank + qk_rope_head_dim]
            input1: cache_lora_kv [batch, maximum kv cache length (prefill len + decode len), kv_lora_rank + qk_rope_head_dim]
            input2: kv_len_list [batch], List[int]. records the history kv cache length of each request in the batch
        """
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.instance = op.instance
        # cache_var_lens, shape [batch], value being the past kv cache length of each batch
        self.cache_var_lens = op.instance.__class__._atencost["var_len_list"]
        self.q_mul_wuk_dtype, self.cache_lora_kv_dtype = self.inputs_dtype
        self._get_op_size()

    def _get_op_size(self):
        q_mul_wuk_shape, cache_lora_kv_shape = self.inputs_shape
        self.batch_times_decode_length, self.head_num, _ = q_mul_wuk_shape
        self.batch_size, self.max_prefill_decode_length, _ = cache_lora_kv_shape
        self.decode_length = self.batch_times_decode_length // self.batch_size
        self.max_prefill_length = self.max_prefill_decode_length - self.decode_length

    def _create_flash_attention_predictors(self):
        if hasattr(self.instance, 'weight_dtype_fp4'):
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
            self.max_prefill_decode_length,
        ]

        attn_scores_shape = MatmulPrediction.infer_shape(
            q_mul_wuk_transpose_shape, cache_lora_kv_transpose
        )

        q_mul_k_op = create_op_info(
            input_list=[q_mul_wuk_transpose_shape, cache_lora_kv_transpose],
            input_dtype_list=[self.instance.weight_dtype, self.instance.weight_dtype],
            output_list=[attn_scores_shape],
            output_dtype_list=[self.instance.weight_dtype],
            other_input_list=[self.cache_var_lens]
        )

        qk_prediction = VarLenQKMatmulPrediction(q_mul_k_op, self.hardware)

        # attn_scores_shape: [decode_len, batch_size, num_heads, max_kv_cache_prefill_decode_len]

        # softmax
        softmax_op = create_op_info(
            input_list=[attn_scores_shape],
            input_dtype_list=[self.instance.softmax_dtype],
            output_list=[attn_scores_shape],  # Softmax does not change shape
            output_dtype_list=[self.instance.softmax_dtype],
            other_input_list=[self.cache_var_lens]
        )

        softmax_predictor = VarLenSoftmaxPrediction(softmax_op, self.hardware)

        # score @ V
        cache_nope_kv_shape = [
            self.batch_size,
            self.max_prefill_decode_length,
            self.instance.kv_lora_rank,
        ]
        out_put_shape = MatmulPrediction.infer_shape(
            attn_scores_shape, cache_nope_kv_shape
        )
        attn_mul_v_op = create_op_info(
            input_list=[attn_scores_shape, cache_nope_kv_shape],
            input_dtype_list=[self.instance.weight_dtype, self.instance.weight_dtype],
            output_list=[out_put_shape],  # Final output shape
            output_dtype_list=[self.instance.output_dtype],
            other_input_list=[self.cache_var_lens]
        )
        attn_v_predictor = VarLenScoreVMatmulPrediction(attn_mul_v_op, self.hardware)

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

        softmax_flops_dict = {
            k: v for k, v in softmax_predictor.vec_flops_dict.items()
        }

        return softmax_flops_dict

    @property
    def op_time(self):
        overlap_softmax = 0.5
        return max([self.cube_time + self.vec_time * overlap_softmax, self.memory_time])
