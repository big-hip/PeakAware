from math import prod
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_costmodel.element_op import Element3Prediction
from zhanlu.backend.analytical_model.op_manager import op_manager
from zhanlu.backend.analytical_model.utils.model_utils import TBps2Bpus, calculate_total_bytes
from zhanlu.backend.perf_result import ZhanluPerfResult
from zhanlu.backend.analytical_model.utils.datatype import get_dtype_size


@op_manager.register("InitRoutingQuant")
class InitRoutingQuantPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.module = op.instance
        self.op_type = "vector"

    @property
    def memory_size(self):
        in_mem_size = calculate_total_bytes(self.inputs_shape, self.inputs_dtype)
        outputs_dtype = self.outputs_dtype
        outputs_dtype[0] = self.module.quant_dtype
        out_mem_size = calculate_total_bytes(self.outputs_shape, outputs_dtype)

        return in_mem_size + out_mem_size

    @property
    def memory_time(self):
        memory_utilization = 0.2
        return self.memory_size / (self.memory_bandwidth * memory_utilization) / TBps2Bpus  # 返回us
