from math import prod
from zhanlu.backend.analytical_model.op_costmodel.element_op import Element4Prediction
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from zhanlu.backend.perf_result import ZhanluPerfResult
from zhanlu.backend.analytical_model.utils.datatype import get_dtype_size


@op_manager.register("FusedSubCastLessFillForGating")
class FusedSubCastLessFillForGatingPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "vector"
