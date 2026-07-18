from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from math import prod

# MulExpAddMul (dt, self.dt_bias, A_log, x, self.head_dim)
@op_manager.register("MulExpAddMul")
class MulExpAddMul(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
           (dt, self.dt_bias, A_log, x, self.head_dim)

        output:
            x_scaled, dA
        '''
        super().__init__(op, hardware)
        self.op = op
        self.op_type = "vector"
        self.instance = op.instance

        # x_flat, B_flat, ssm_state, dA, C, D, x = self.inputs_shape
        self.dt, self.dt_bias, self.A_log, self.x = self.inputs_shape
        self.vector_dtype = op.instance.vector_dtype

    @property
    def vec_flops_dict(self):
        flops = 0
        # flops+=1
        # Exp and Neg
        flops += 2 * prod(self.A_log)
        # Add
        flops += prod(self.dt)
        #Mul
        flops += prod(self.dt)
        #Exp
        flops += prod(self.dt)
        #Mul
        flops += prod(self.x)

        return {str(self.vector_dtype): flops}