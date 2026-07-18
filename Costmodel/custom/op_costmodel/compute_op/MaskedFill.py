from math import prod
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_costmodel.element_op import Element3Prediction
from zhanlu.backend.analytical_model.op_manager import op_manager
from zhanlu.backend.analytical_model.utils.model_utils import TBps2Bpus

# TODO 暂时没用到 加个1规避一下
@op_manager.register("MaskedFill1")
class MaskedFillPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "vector"
        self.module = op.instance
