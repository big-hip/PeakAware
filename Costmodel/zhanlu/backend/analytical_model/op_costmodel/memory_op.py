from numpy import dtype
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager


@op_manager.register("Concat", "Slice", "SliceBackward", "T", "SplitWithSizes", "Cat", "NllLossBackward", "Split", "Expand",
                     "ScatterAdd", "ScatterInplace", "Sin", "Cos", "ZerosLike", "Histc", "Gather", "Permute", "Cumsum",
                     "NpuMoeTokenUnpermuteGrad", "NpuMoeTokenPermuteGrad",
                     "Embedding", "EmbeddingDenseBackward", "Transpose", "CopyInplace", "ToCopy",
                     "NpuMoeTokenUnpermute", "View", "UnsafeView", "Reshape", "Clone", "Detach", "Select",
                     "SelectBackward", "Unsqueeze", "Squeeze", "Getitem", "MaxPool2dWithIndices",
                     "MaxPool2dWithIndicesBackward")
class MemoryPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
