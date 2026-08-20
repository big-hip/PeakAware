from math import prod
from custom.op_costmodel.compute_op.group_matmul import get_prediction_by_gmm

from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_costmodel.element_op import Element3Prediction
from atencost.backend.analytical_model.op_manager import op_manager

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


@op_manager.register("GroupedLinear")
class GroupedLinearPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.module = op.instance
        self.op_type = "cube"
        self.flops_ratio = 1 / self.module.ep_size * dynamic_workload_ratio[self.module.ep_size]
        self.weight = self.module.weight
        self.weight_dtype = self.module.weight_dtype

    def _create_gl_predictors(self):

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
            self.hardware,  # originally no hardware here
        )

        return h_down_prediction

    @property
    def memory_size(self):
        h_down_prediction = self._create_gl_predictors()

        mm_mem = h_down_prediction.memory_size

        return mm_mem

    @property
    def cube_flops_dict(self):
        input_shape = self.inputs_shape[0]
        input_dtype = self.inputs_dtype[0]

        flops = self.flops_ratio * 2 * prod(input_shape) * self.module.out_features
        return {input_dtype: flops}
