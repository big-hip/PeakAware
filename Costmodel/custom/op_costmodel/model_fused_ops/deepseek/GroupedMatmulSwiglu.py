from math import prod
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_costmodel.element_op import Element3Prediction, Element5Prediction
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.analytical_model.utils.helper import broadcast_shapes, create_op_info, merge_flops_dicts

from custom.op_costmodel.compute_op.group_matmul import get_prediction_by_gmm

dynamic_workload_ratio = {
    1: 1.0,
    2: 1.000084975670124,
    4: 1.0004471088277882,
    8: 1.001592701879041,
    16: 1.0066749309671337,
    32: 1.038354281721444,
    64: 1.4276459792564655,
    128: 2.7399397225215516,
    256: 5.4264547413793105,
}


@op_manager.register("GroupedMatmulSwiglu")
class GroupedMatmulSwigluPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            h (Tensor)
        """
        super().__init__(op, hardware)
        self.op_type = "mix"
        self.module = op.instance
        self.flops_ratio = 1 / self.module.ep_size * dynamic_workload_ratio[self.module.ep_size]

    def _create_gmms_predictors(self):

        h_shape = self.inputs_shape[0]
        h_dtype = self.inputs_dtype[0]

        # down
        sharded_weight_shape = self.module.weight_shape
        group_num, out_features, _ = sharded_weight_shape
        h_down_prediction, h_down_shape, h_down_dtype = get_prediction_by_gmm(
            h_shape,
            h_dtype,
            group_num,
            out_features,
            self.module.weight_dtype,
            self.hardware,  # originally, there is no self.hardware added here
        )

        # swiglu
        swiglu_h_shape = h_down_shape.copy()
        swiglu_h_shape[-1] = swiglu_h_shape[-1] // 2

        swiglu_op = create_op_info(
            input_list=[h_down_shape],
            input_dtype_list=[self.module.vector_dtype],
            output_list=[swiglu_h_shape],
            output_dtype_list=[self.module.vector_dtype],
        )

        swiglu_prediction = Element5Prediction(swiglu_op, self.hardware)

        return h_down_prediction, swiglu_prediction

    @property
    def cube_flops_dict(self):
        h_down_prediction, swiglu_prediction = self._create_gmms_predictors()

        flops_1 = {k: (v * self.flops_ratio) for k, v in h_down_prediction.cube_flops_dict.items()}
        total_flops_dict = merge_flops_dicts([flops_1])

        return total_flops_dict

    @property
    def cube_time_dict(self):
        h_down_prediction, swiglu_prediction = self._create_gmms_predictors()

        time_1 = {k: v for k, v in h_down_prediction.cube_time_dict.items()}
        total_time_dict = merge_flops_dicts([time_1])

        return total_time_dict

    @property
    def vec_flops_dict(self):
        _, swiglu_prediction = self._create_gmms_predictors()

        swiglu_flops = {k: v / 2 for k, v in swiglu_prediction.vec_flops_dict.items()}

        total_flops_dict = merge_flops_dicts([swiglu_flops])

        return total_flops_dict

    @property
    def vec_flops_time(self):
        _, swiglu_prediction = self._create_gmms_predictors()

        swiglu_time = {k: v for k, v in swiglu_prediction.vec_flops_time.items()}

        total_time_dict = merge_flops_dicts([swiglu_time])

        return total_time_dict

    @property
    def memory_size(self):
        h_down_prediction, swiglu_prediction = self._create_gmms_predictors()

        # calculate memory access for matmul input and weight
        mm_in_mem = h_down_prediction.input_memory_size

        # with fused swiglu, matmul output copy is skipped, consider swiglu output only
        swiglu_out_mem = swiglu_prediction.output_memory_size

        return mm_in_mem + swiglu_out_mem
