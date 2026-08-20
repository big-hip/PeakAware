from math import prod
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_costmodel.element_op import Element3Prediction
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.analytical_model.utils.model_utils import TBps2Bpus
from atencost.backend.perf_result import OpPerfResult
from atencost.backend.analytical_model.utils.datatype import get_dtype_size

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


@op_manager.register("MoETokenUnpermute")
class MoETokenUnpermutePrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "vector"
        self.module = op.instance
        self.hidden = self.module.hidden
        self.ep_size = self.module.ep_size
        self.mem_access_ratio = 1 / self.ep_size * dynamic_workload_ratio[self.ep_size]

    @property
    def memory_size(self):
        gemm_out_shape, _, _, num_tokens_shape = self.inputs_shape
        gemm_out_dtype, _, _, _ = self.inputs_dtype

        input_mem = prod(gemm_out_shape) * get_dtype_size(gemm_out_dtype)
        input_mem = self.mem_access_ratio * input_mem
        output_mem = num_tokens_shape[0] * self.hidden * get_dtype_size(gemm_out_dtype)

        return input_mem + output_mem


@op_manager.register(
    "npu_moe_token_unpermute_forward",
    "npu_moe_token_permute_forward",
    "npu_moe_token_unpermute_gard",
    "npu_moe_token_permute_grad",
    "NpuMoeTokenPermute"
)
class NpuMoETokenUnpermutePrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "vector"
