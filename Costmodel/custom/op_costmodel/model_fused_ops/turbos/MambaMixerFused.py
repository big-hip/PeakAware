from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager

from math import prod


@op_manager.register("MambaMixerFused")
class MambaMixerFusedPrediction(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
            x_flat (B*H, P, 1)
            B_flat (B*H, 1, N)

            ssm_state (B, H*P, N) -> (B, H, P, N)
            dA (B, H) -> dA (B, H, 1, 1)

            C (B, 1, G*N) -> C_expanded (B, H, N)
            D (H) -> D_expanded (1, H, 1)

            x (B, H, P)

        output:
            y (B, H, P)
            ssm_state (B, H, P, N)

        timeline (ignore reshape)
            dBx = bmm(x_flat, B_flat)
            ssm_state = ssm_state * dA + dBx
            y = (ssm_state * C_expanded).sum(-1)
            y = y + D_expanded * x
        '''
        super().__init__(op, hardware)
        self.op = op
        self.op_type = "mix"
        self.instance = op.instance

        x_flat, B_flat, ssm_state, dA, C, D, x = self.inputs_shape
        self.b, self.h, self.p = x
        _, _, self.n = ssm_state

        x_flat_dtype = self.inputs_dtype[0]
        self.bmm_dtype = x_flat_dtype

        self.vector_dtype = op.instance.vector_dtype


    @property
    def cube_flops_dict(self):
        flops = 2 * self.b * self.h * self.p * self.n
        return {self.bmm_dtype: flops}

    @property
    def vec_flops_dict(self):
        flops = 0
        # ssm_state = ssm_state * dA + dBx
        flops += 2 * self.b * self.h * self.p * self.n

        # y = (ssm_state * C_expanded).sum(-1)
        flops += 2 * self.b * self.h * self.p * self.n

        # y = y + D_expanded * x
        flops += 2 * self.b * self.h * self.p

        return {str(self.vector_dtype): flops}