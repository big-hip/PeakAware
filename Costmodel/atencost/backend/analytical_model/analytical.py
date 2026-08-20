from atencost.backend.base_model import BaseModel
from atencost.backend.perf_result import OpPerfResult
from atencost.backend.analytical_model.op_manager import OpManager
import warnings

warnings.filterwarnings("once", category=DeprecationWarning)

class AnalyticalModel(BaseModel):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.model_manager = OpManager()

    def __call__(self) -> OpPerfResult:
        if self.op.name in self.model_manager.registry.keys():
            self.result = self.model_manager.predict_single_op_pref(self.op.name, self.op, self.hardware)
        else:
            warnings.warn(f"{self.op.name} is missed", DeprecationWarning, stacklevel=2)

        return self.result
