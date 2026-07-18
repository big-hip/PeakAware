from numpy import dtype

from custom.op_costmodel.compute_op.matmul_einsum import MatmulEinsumPrediction

from zhanlu.backend.analytical_model.op_costmodel.element_op import (
    Element3Prediction,
    Element4Prediction,
)
from custom.op_costmodel.compute_op.RmsNormQuant import (
    RmsNormQuantPrediction,
    QuantPrediction
)
from zhanlu.backend.analytical_model.op_costmodel.matmul import (
    MatmulPrediction,
    get_prediction_by_linear,
)
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager

from zhanlu.backend.analytical_model.utils.helper import (
    broadcast_shapes,
    create_op_info,
    merge_flops_dicts,
)
from math import prod
import torch


@op_manager.register("MLAPreprocessKernel")
class MLAPreprocessPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        h :[batch_times_decode_length, hidden_size]
        cache_lora_kv: [batch, prefill_length, kv_lora_rank + qk_rope_head_dim]
        """
        super().__init__(op, hardware)
        self.instance = op.instance
        self.h_dtype, self.cache_k_dtype = self.inputs_dtype
        self._get_op_size()
        if "head_tail_time_for_fused_op" in self.hardware_config["chip_config"].keys():
            self.head_tail_time = self.hardware_config["chip_config"]["head_tail_time_for_fused_op"] * 10
        chip_name, _ = hardware.split(',')
        self.quant_flag = hasattr(self.instance, "weight_dtype_fp4")
        if chip_name == 'A2' or chip_name == 'A3':
            self.quant_flag = self.quant_flag or self.instance.weight_dtype == torch.float8_e4m3fn or self.instance.weight_dtype == torch.float8_e5m2

    def _get_op_size(self):
        self.h_shape, self.cache_k_shape = self.inputs_shape
        self.batch_times_decode_length, _ = self.h_shape
        self.batch_size, self.prefill_length, _ = self.cache_k_shape
        if self.batch_times_decode_length % self.batch_size != 0:
            raise ValueError(
                f"batch_times_decode_length {self.batch_times_decode_length} is not divisible by batch_size {self.batch_size}"
            )
        self.decode_length = self.batch_times_decode_length // self.batch_size

    def _create_flash_attention_predictors(self):
        # TODO 现在获取vector的prediction需要手动找到对应的类
        # rmsnorm_h
        if hasattr(self.instance, "weight_dtype_fp4"):
            self.instance.weight_dtype = self.instance.weight_dtype_fp4

        h_shape, cache_k_shape = self.h_shape, self.cache_k_shape

        rmsnorm_h_op = create_op_info(
            input_list=[h_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[h_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        if self.quant_flag:
            rmsnorm_h_prediction = RmsNormQuantPrediction(rmsnorm_h_op, self.hardware)
            self.h_dtype = self.instance.weight_dtype
        else:
            rmsnorm_h_prediction = Element4Prediction(rmsnorm_h_op, self.hardware)

        # wq_a
        wq_a_prediction, q_lora_shape, q_lora_dtype = get_prediction_by_linear(
            h_shape, self.h_dtype, self.instance.wq_a, self.hardware, self.instance.weight_dtype
        )
        if self.quant_flag:
            self.h_dtype = self.instance.vector_dtype

        # rmsnorm_q_lora
        rmsnorm_q_lora_op = create_op_info(
            input_list=[q_lora_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[q_lora_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        if self.quant_flag:
            rmsnorm_q_lora_prediction = RmsNormQuantPrediction(rmsnorm_q_lora_op, self.hardware)
            self.h_dtype = self.instance.weight_dtype
        else:
            rmsnorm_q_lora_prediction = Element4Prediction(rmsnorm_q_lora_op, self.hardware)

        # wq_b
        wq_b_prediction, q_up_shape, q_up_dtype = get_prediction_by_linear(
            q_lora_shape, self.h_dtype, self.instance.wq_b, self.hardware, self.instance.weight_dtype
        )
        if self.quant_flag:
            self.h_dtype = self.instance.vector_dtype

        # rope_q
        # 省略split
        head_num = q_up_shape[1] // (self.instance.qk_nope_head_dim + self.instance.qk_rope_head_dim)
        q_nope_shape = [q_up_shape[0], head_num, self.instance.qk_nope_head_dim]
        q_for_rope_shape = [q_up_shape[0], head_num, self.instance.qk_rope_head_dim]

        rope_q_op = create_op_info(
            input_list=[q_for_rope_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[q_for_rope_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rope_q_prediction = Element3Prediction(rope_q_op, self.hardware)

        # w_uk
        equation = "bij,ijk->bik"
        w_uk_shape = [
            head_num,
            self.instance.qk_nope_head_dim,
            self.instance.kv_lora_rank,
        ]
        q_nope_mul_wuk_shape = MatmulEinsumPrediction.infer_shape(equation, q_nope_shape, w_uk_shape)

        w_uk_op = create_op_info(
            input_list=[q_nope_shape, w_uk_shape],
            input_dtype_list=[self.h_dtype, self.instance.weight_dtype],
            output_list=[q_nope_mul_wuk_shape],
            output_dtype_list=[self.h_dtype],
            other_input_list=[equation],
        )
        w_uk_prediction = MatmulEinsumPrediction(w_uk_op, self.hardware)

        # 省略concat q
        q_mul_wuk = [
            q_nope_mul_wuk_shape[0],
            head_num,
            q_nope_mul_wuk_shape[2] + q_for_rope_shape[2],
        ]

        # wkv_a
        wkv_a_prediction, kv_lora_rope_shape, kv_lora_rope_dtype = get_prediction_by_linear(
            h_shape, self.h_dtype, self.instance.wkv_a, self.hardware, self.instance.weight_dtype
        )

        # 省略split
        kv_lora_shape = [kv_lora_rope_shape[0], self.instance.kv_lora_rank]
        k_for_rope_shape = [kv_lora_rope_shape[0], self.instance.qk_rope_head_dim]

        # rope_k
        rope_k_op = create_op_info(
            input_list=[k_for_rope_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[k_for_rope_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rope_k_prediction = Element3Prediction(rope_k_op, self.hardware)

        # rmsnorm_kv_lora
        rmsnorm_kv_lora_op = create_op_info(
            input_list=[kv_lora_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[kv_lora_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rmsnorm_kv_lora_prediction = Element4Prediction(rmsnorm_kv_lora_op, self.hardware)

        q_shape = [q_up_shape[0], head_num, self.instance.qk_nope_head_dim + self.instance.qk_rope_head_dim]
        quant_q_op = create_op_info(
            input_list=[q_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[q_shape],
            output_dtype_list=[self.h_dtype],
        )
        quant_q_prediction = QuantPrediction(quant_q_op, self.hardware)

        kv_shape = [kv_lora_rope_shape[0], self.instance.kv_lora_rank + self.instance.qk_rope_head_dim]

        quant_kv_op = create_op_info(
            input_list=[kv_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[kv_shape],
            output_dtype_list=[self.h_dtype],
        )
        quant_kv_prediction = QuantPrediction(quant_kv_op, self.hardware)

        return (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        )

    @property
    def cube_flops_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        ) = self._create_flash_attention_predictors()

        wq_a_flops = {k: v for k, v in wq_a_prediction.cube_flops_dict.items()}
        wq_b_flops = {k: v for k, v in wq_b_prediction.cube_flops_dict.items()}
        wuk_flops = {k: v for k, v in w_uk_prediction.cube_flops_dict.items()}
        wkv_a_flops = {k: v for k, v in wkv_a_prediction.cube_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([wq_a_flops, wq_b_flops, wuk_flops, wkv_a_flops])

        return total_flops_dict

    @property
    def cube_time_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        ) = self._create_flash_attention_predictors()

        wq_a_time = {k: v for k, v in wq_a_prediction.cube_time_dict.items()}
        wq_b_time = {k: v for k, v in wq_b_prediction.cube_time_dict.items()}
        wuk_time = {k: v for k, v in w_uk_prediction.cube_time_dict.items()}
        wkv_a_time = {k: v for k, v in wkv_a_prediction.cube_time_dict.items()}

        total_flops_dict = merge_flops_dicts([wq_a_time, wq_b_time, wuk_time, wkv_a_time])

        return total_flops_dict

    @property
    def vec_flops_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        ) = self._create_flash_attention_predictors()

        rmsnorm_h_flops = {k: v for k, v in rmsnorm_h_prediction.vec_flops_dict.items()}
        rmsnorm_q_flops = {k: v for k, v in rmsnorm_q_lora_prediction.vec_flops_dict.items()}
        rope_q_flops = {k: v for k, v in rope_q_prediction.vec_flops_dict.items()}
        rope_k_flops = {k: v for k, v in rope_k_prediction.vec_flops_dict.items()}
        rmsnorm_kv_lora_flops = {k: v for k, v in rmsnorm_kv_lora_prediction.vec_flops_dict.items()}
        flops_list = [rmsnorm_h_flops, rmsnorm_q_flops, rope_q_flops, rope_k_flops, rmsnorm_kv_lora_flops]
        if self.quant_flag:
            quant_q_flops = {k: v for k, v in quant_q_prediction.vec_flops_dict.items()}
            flops_list.append(quant_q_flops)
            quant_kv_flops = {k: v for k, v in quant_kv_prediction.vec_flops_dict.items()}
            flops_list.append(quant_kv_flops)

        total_flops_dict = merge_flops_dicts(flops_list)

        return total_flops_dict

    @property
    def vec_time_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        ) = self._create_flash_attention_predictors()

        rmsnorm_h_time = {k: v for k, v in rmsnorm_h_prediction.vec_time_dict.items()}
        rmsnorm_q_time = {k: v for k, v in rmsnorm_q_lora_prediction.vec_time_dict.items()}
        rope_q_time = {k: v for k, v in rope_q_prediction.vec_time_dict.items()}
        rope_k_time = {k: v for k, v in rope_k_prediction.vec_time_dict.items()}
        rmsnorm_kv_lora_time = {k: v for k, v in rmsnorm_kv_lora_prediction.vec_time_dict.items()}
        time_list = [rmsnorm_h_time, rmsnorm_q_time, rope_q_time, rope_k_time, rmsnorm_kv_lora_time]
        if self.quant_flag:
            quant_q_time = {k: v for k, v in quant_q_prediction.vec_time_dict.items()}
            time_list.append(quant_q_time)
            quant_kv_time = {k: v for k, v in quant_kv_prediction.vec_time_dict.items()}
            time_list.append(quant_kv_time)

        total_time_dict = merge_flops_dicts(time_list)

        return total_time_dict

    @property
    def memory_size(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        ) = self._create_flash_attention_predictors()

        mem = wq_a_prediction.memory_size
        mem += wq_b_prediction.memory_size
        mem += w_uk_prediction.memory_size
        mem += wkv_a_prediction.memory_size

        return mem

    @property
    def memory_time(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
            quant_q_prediction,
            quant_kv_prediction
        ) = self._create_flash_attention_predictors()

        mem = wq_a_prediction.memory_time
        mem += wq_b_prediction.memory_time
        mem += w_uk_prediction.memory_time
        mem += wkv_a_prediction.memory_time

        return mem


@op_manager.register("MLAPreProcess_KV")
class MLAPreProcessKVPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        h :[batch_times_decode_length, hidden_size]
        cache_lora_kv: [batch, prefill_length, kv_lora_rank + qk_rope_head_dim]
        """
        super().__init__(op, hardware)
        self.instance = op.instance
        self.h_dtype, self.cache_k_dtype = self.inputs_dtype
        self._get_op_size()
        if "head_tail_time_for_fused_op" in self.hardware_config["chip_config"].keys():
            self.head_tail_time = self.hardware_config["chip_config"]["head_tail_time_for_fused_op"] * 4

    def _get_op_size(self):
        self.h_shape, self.cache_k_shape = self.inputs_shape
        self.batch_times_decode_length, _ = self.h_shape
        self.batch_size, self.prefill_length, _ = self.cache_k_shape
        self.decode_length = self.batch_times_decode_length // self.batch_size

    def _create_flash_attention_predictors(self):
        # TODO 现在获取vector的prediction需要手动找到对应的类
        # rmsnorm_h
        if hasattr(self.instance, "weight_dtype_fp4"):
            self.instance.weight_dtype = self.instance.weight_dtype_fp4
        h_shape, cache_k_shape = self.h_shape, self.cache_k_shape

        rmsnorm_h_op = create_op_info(
            input_list=[h_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[h_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rmsnorm_h_prediction = Element4Prediction(rmsnorm_h_op, self.hardware)

        # wq_a
        wq_a_prediction, q_lora_shape, q_lora_dtype = get_prediction_by_linear(
            h_shape, self.h_dtype, self.instance.wq_a, self.hardware, self.instance.weight_dtype
        )

        # rmsnorm_q_lora
        rmsnorm_q_lora_op = create_op_info(
            input_list=[q_lora_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[q_lora_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rmsnorm_q_lora_prediction = RmsNormQuantPrediction(rmsnorm_q_lora_op, self.hardware)

        # wkv_a
        wkv_a_prediction, kv_lora_rope_shape, kv_lora_rope_dtype = get_prediction_by_linear(
            h_shape, self.h_dtype, self.instance.wkv_a, self.hardware, self.instance.weight_dtype
        )

        # 省略split
        kv_lora_shape = [kv_lora_rope_shape[0], self.instance.kv_lora_rank]
        k_for_rope_shape = [kv_lora_rope_shape[0], self.instance.qk_rope_head_dim]

        # rope_k
        rope_k_op = create_op_info(
            input_list=[k_for_rope_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[k_for_rope_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rope_k_prediction = Element3Prediction(rope_k_op, self.hardware)

        # rmsnorm_kv_lora
        rmsnorm_kv_lora_op = create_op_info(
            input_list=[kv_lora_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[kv_lora_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rmsnorm_kv_lora_prediction = Element4Prediction(rmsnorm_kv_lora_op, self.hardware)

        return (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        )

    @property
    def cube_flops_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        ) = self._create_flash_attention_predictors()

        wq_a_flops = {k: v for k, v in wq_a_prediction.cube_flops_dict.items()}
        wkv_a_flops = {k: v for k, v in wkv_a_prediction.cube_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([wq_a_flops, wkv_a_flops])

        return total_flops_dict

    @property
    def cube_time_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        ) = self._create_flash_attention_predictors()

        wq_a_time = {k: v for k, v in wq_a_prediction.cube_time_dict.items()}
        wkv_a_time = {k: v for k, v in wkv_a_prediction.cube_time_dict.items()}

        total_flops_dict = merge_flops_dicts([wq_a_time, wkv_a_time])

        return total_flops_dict

    @property
    def vec_flops_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        ) = self._create_flash_attention_predictors()

        rmsnorm_h_flops = {k: v for k, v in rmsnorm_h_prediction.vec_flops_dict.items()}
        rmsnorm_q_flops = {k: v for k, v in rmsnorm_q_lora_prediction.vec_flops_dict.items()}
        rope_k_flops = {k: v for k, v in rope_k_prediction.vec_flops_dict.items()}
        rmsnorm_kv_lora_flops = {k: v for k, v in rmsnorm_kv_lora_prediction.vec_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([rmsnorm_h_flops, rmsnorm_q_flops, rope_k_flops, rmsnorm_kv_lora_flops])

        return total_flops_dict

    @property
    def vec_time_dict(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        ) = self._create_flash_attention_predictors()

        rmsnorm_h_time = {k: v for k, v in rmsnorm_h_prediction.vec_time_dict.items()}
        rmsnorm_q_time = {k: v for k, v in rmsnorm_q_lora_prediction.vec_time_dict.items()}
        rope_k_time = {k: v for k, v in rope_k_prediction.vec_time_dict.items()}
        rmsnorm_kv_lora_time = {k: v for k, v in rmsnorm_kv_lora_prediction.vec_time_dict.items()}

        total_time_dict = merge_flops_dicts([rmsnorm_h_time, rmsnorm_q_time, rope_k_time, rmsnorm_kv_lora_time])

        return total_time_dict

    @property
    def memory_size(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        ) = self._create_flash_attention_predictors()

        mem = wq_a_prediction.memory_size
        mem += wkv_a_prediction.memory_size

        return mem

    @property
    def memory_time(self):
        (
            rmsnorm_h_prediction,
            wq_a_prediction,
            rmsnorm_q_lora_prediction,
            wkv_a_prediction,
            rope_k_prediction,
            rmsnorm_kv_lora_prediction,
        ) = self._create_flash_attention_predictors()

        mem = wq_a_prediction.memory_time
        mem += wkv_a_prediction.memory_time

        return mem


@op_manager.register("MLAPreProcess_Q")
class MLAPreprocessQPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        rmsnorm_q_lora : [batch_times_decode_length, q_lora_rank]
        """
        super().__init__(op, hardware)
        self.instance = op.instance
        self.rmsnorm_q_lora_dtype = self.inputs_dtype[0]
        self._get_op_size()
        if "head_tail_time_for_fused_op" in self.hardware_config["chip_config"].keys():
            self.head_tail_time = self.hardware_config["chip_config"]["head_tail_time_for_fused_op"] * 4

    def _get_op_size(self):
        self.rmsnorm_q_lora_shape = self.inputs_shape[0]
        self.batch_times_decode_length, _ = self.rmsnorm_q_lora_shape

    def _create_flash_attention_predictors(self):
        if hasattr(self.instance, "weight_dtype_fp4"):
            self.instance.weight_dtype = self.instance.weight_dtype_fp4
        rmsnorm_q_lora_shape = self.rmsnorm_q_lora_shape

        # wq_b
        wq_b_prediction, q_up_shape, q_up_dtype = get_prediction_by_linear(
            rmsnorm_q_lora_shape,
            self.rmsnorm_q_lora_dtype,
            self.instance.wq_b,
            self.hardware,
            self.instance.weight_dtype,
        )

        # rope_q
        # 省略split
        head_num = q_up_shape[1] // (self.instance.qk_nope_head_dim + self.instance.qk_rope_head_dim)
        q_nope_shape = [q_up_shape[0], head_num, self.instance.qk_nope_head_dim]
        q_for_rope_shape = [q_up_shape[0], head_num, self.instance.qk_rope_head_dim]

        rope_q_op = create_op_info(
            input_list=[q_for_rope_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[q_for_rope_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        rope_q_prediction = Element3Prediction(rope_q_op, self.hardware)

        # w_uk
        equation = "bij,ijk->bik"
        w_uk_shape = [
            head_num,
            self.instance.qk_nope_head_dim,
            self.instance.kv_lora_rank,
        ]
        q_nope_mul_wuk_shape = MatmulEinsumPrediction.infer_shape(equation, q_nope_shape, w_uk_shape)

        w_uk_op = create_op_info(
            input_list=[q_nope_shape, w_uk_shape],
            input_dtype_list=[self.rmsnorm_q_lora_dtype, self.instance.weight_dtype],
            output_list=[q_nope_mul_wuk_shape],
            output_dtype_list=[self.rmsnorm_q_lora_dtype],
            other_input_list=[equation],
        )

        w_uk_prediction = MatmulEinsumPrediction(w_uk_op, self.hardware)

        # 省略concat q
        q_mul_wuk = [
            q_nope_mul_wuk_shape[0],
            head_num,
            q_nope_mul_wuk_shape[2] + q_for_rope_shape[2],
        ]

        return (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        )

    @property
    def cube_flops_dict(self):
        (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        ) = self._create_flash_attention_predictors()

        wq_b_flops = {k: v for k, v in wq_b_prediction.cube_flops_dict.items()}
        wuk_flops = {k: v for k, v in w_uk_prediction.cube_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([wq_b_flops, wuk_flops])

        return total_flops_dict

    @property
    def cube_time_dict(self):
        (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        ) = self._create_flash_attention_predictors()

        wq_b_time = {k: v for k, v in wq_b_prediction.cube_time_dict.items()}
        wuk_time = {k: v for k, v in w_uk_prediction.cube_time_dict.items()}

        total_flops_dict = merge_flops_dicts([wq_b_time, wuk_time])

        return total_flops_dict

    @property
    def vec_flops_dict(self):
        (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        ) = self._create_flash_attention_predictors()

        rope_q_flops = {k: v for k, v in rope_q_prediction.vec_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([rope_q_flops])

        return total_flops_dict

    @property
    def vec_time_dict(self):
        (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        ) = self._create_flash_attention_predictors()

        rope_q_time = {k: v for k, v in rope_q_prediction.vec_time_dict.items()}

        total_time_dict = merge_flops_dicts([rope_q_time])

        return total_time_dict

    @property
    def memory_size(self):
        (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        ) = self._create_flash_attention_predictors()

        mem = wq_b_prediction.memory_size
        mem += w_uk_prediction.memory_size

        return mem

    @property
    def memory_time(self):
        (
            wq_b_prediction,
            rope_q_prediction,
            w_uk_prediction,
        ) = self._create_flash_attention_predictors()

        mem = wq_b_prediction.memory_time
        mem += w_uk_prediction.memory_time

        return mem
