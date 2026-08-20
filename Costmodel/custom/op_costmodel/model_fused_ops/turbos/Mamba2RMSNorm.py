from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager

from math import prod


@op_manager.register("Mamba2RMSNorm")
class Mamba2RMSNormPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            h (Tensor)
        """
        super().__init__(op, hardware)
        self.op = op
        self.op_type = "vector"
        self.instance = op.instance

    @property
    def vec_flops_dict(self):
        # 暂时假设up_cast = False
        vec_flops = 0
        if len(self.inputs_shape) == 2:
            # z is not none
            x_shape, z_shape = self.inputs_shape
            # silu and mul
            vec_flops += prod(z_shape) * 5
        else:
            x_shape = self.inputs_shape[0]
        x_dtype = self.inputs_dtype[0]

        if self.instance.group_size is None:
            vec_flops += prod(x_shape) * 4

        else:
            vec_flops += prod(x_shape) * 4 + prod(x_shape) // self.instance.group_size * 3

        if self.instance.bias is not None:
            vec_flops += prod(x_shape)

        return {x_dtype: vec_flops}