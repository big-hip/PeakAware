from math import prod
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_costmodel.element_op import Element3Prediction
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.analytical_model.utils.model_utils import TBps2Bpus
from atencost.backend.perf_result import OpPerfResult
from atencost.backend.analytical_model.utils.datatype import get_dtype_size


@op_manager.register("FusedAddTopkDivKernel")
class FusedAddTopkDivKernelPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "vector"
        self.module = op.instance
        self.output_dtype = self.module.output_dtype
        self.n_routed_experts = self.module.n_routed_experts
        self.num_experts_per_tok = self.module.num_experts_per_tok

    @property
    def vec_flops_dict(self):
        input_a_shape = self.inputs_shape[0]
        input_a_dtype = self.inputs_dtype[0]
        flops = prod(input_a_shape)

        return {input_a_dtype: flops}

    @property
    def memory_size(self):
        input_a_shape = self.inputs_shape[0]
        input_a_dtype = self.inputs_dtype[0]

        input_a_memory_access = prod(input_a_shape) * get_dtype_size(input_a_dtype)
        weight_memory_access = self.n_routed_experts * get_dtype_size(self.output_dtype)
        output_memory_access = input_a_shape[0] * self.num_experts_per_tok * get_dtype_size(input_a_dtype)

        return input_a_memory_access + weight_memory_access + output_memory_access
