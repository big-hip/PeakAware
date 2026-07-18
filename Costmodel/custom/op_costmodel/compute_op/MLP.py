from zhanlu.backend.analytical_model.op_costmodel.element_op import (
    Element5Prediction,
)
from zhanlu.backend.analytical_model.op_costmodel.matmul import (
    get_prediction_by_linear,
)
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager

from zhanlu.backend.analytical_model.utils.helper import (
    broadcast_shapes,
    create_op_info,
    merge_flops_dicts,
)


@op_manager.register("MLP")
class MlpPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            x (Tensor)
        """
        super().__init__(op, hardware)
        self.op_type = "mix"
        self.instance = op.instance

    def _create_flash_attention_predictors(self):

        h = self.inputs_shape[0]
        h_dtype = self.inputs_dtype[0]
        if hasattr(self.instance, "weight_dtype_fp4"):
            weight_dtype = self.instance.weight_dtype_fp4
        else:
            weight_dtype = None

        # up
        h_up_prediction, h_up_shape, h_up_dtype = get_prediction_by_linear(
            h, h_dtype, self.instance.w1, self.hardware, weight_dtype
        )

        # swiglu
        swiglu_h_shape = h_up_shape.copy()
        swiglu_h_shape[-1] = swiglu_h_shape[-1] // 2

        swiglu_op = create_op_info(
            input_list=[h_up_shape],
            input_dtype_list=[self.instance.vector_dtype],
            output_list=[swiglu_h_shape],
            output_dtype_list=[self.instance.vector_dtype],
        )

        swiglu_prediction = Element5Prediction(swiglu_op, self.hardware)

        # down
        h_down_prediction, h_down_shape, h_down_dtype = get_prediction_by_linear(
            swiglu_h_shape, h_dtype, self.instance.w2, self.hardware, weight_dtype
        )

        return h_up_prediction, swiglu_prediction, h_down_prediction

    @property
    def cube_flops_dict(self):
        h_up_prediction, swiglu_prediction, h_down_prediction = self._create_flash_attention_predictors()

        flops_1 = {k: v for k, v in h_up_prediction.cube_flops_dict.items()}
        flops_2 = {k: v for k, v in h_down_prediction.cube_flops_dict.items()}
        total_flops_dict = merge_flops_dicts([flops_1, flops_2])

        return total_flops_dict

    @property
    def cube_time_dict(self):
        h_up_prediction, swiglu_prediction, h_down_prediction = self._create_flash_attention_predictors()

        time_1 = {k: v for k, v in h_up_prediction.cube_time_dict.items()}
        time_2 = {k: v for k, v in h_down_prediction.cube_time_dict.items()}
        total_time_dict = merge_flops_dicts([time_1, time_2])
        return total_time_dict

    @property
    def vec_flops_dict(self):
        h_up_prediction, swiglu_prediction, h_down_prediction = self._create_flash_attention_predictors()

        swiglu_flops = {k: v for k, v in swiglu_prediction.vec_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([swiglu_flops])

        return total_flops_dict

    @property
    def vec_flops_time(self):
        h_up_prediction, swiglu_prediction, h_down_prediction = self._create_flash_attention_predictors()

        swiglu_time = {k: v for k, v in swiglu_prediction.vec_flops_time.items()}

        total_time_dict = merge_flops_dicts([swiglu_time])

        return total_time_dict

    @property
    def memory_size(self):
        h_up_prediction, swiglu_prediction, h_down_prediction = self._create_flash_attention_predictors()

        m_1 = h_up_prediction.memory_size
        m_2 = h_down_prediction.memory_size
        return m_1 + m_2

    @property
    def memory_time(self):
        h_up_prediction, swiglu_prediction, h_down_prediction = self._create_flash_attention_predictors()

        m_1 = h_up_prediction.memory_time
        m_2 = h_down_prediction.memory_time
        return m_1 + m_2
