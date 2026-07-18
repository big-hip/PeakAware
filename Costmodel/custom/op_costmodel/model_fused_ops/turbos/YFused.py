from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from math import prod

# YFused(D, x_pad, A_cum, decay, states, C_b, Y_diag)
@op_manager.register("YFused")
class YFused(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
			D, x_pad, A_cum, decay, states, C_b, Y_diag

        output:
			y
        '''
        super().__init__(op, hardware)
        self.op = op
        self.op_type = "mix"
        self.instance = op.instance

        # x_flat, B_flat, ssm_state, dA, C, D, x = self.inputs_shape
        self.D, self.x_pad , self.A_cum, self.decay, self.states, self.C_b, self.Y_diag = self.inputs_shape
        self.bmm_dtype =  self.inputs_dtype[-1]
        self.vector_dtype = op.instance.vector_dtype


    @property
    def cube_flops_dict(self):
        flops_bmm1 = 2 * prod(self.states) * self.states[1]
        #16 2 64 (128 128 )-> 1024 2 16384 = 2* 1024 * 2 * 2* 16384
        flops_bmm2 = 2 * prod(self.C_b) * self.C_b[-1]
        return {self.bmm_dtype: flops_bmm1 + flops_bmm2}

    @property
    def vec_flops_dict(self):
        flops = 0

        # MUL
        flops += prod(self.x_pad)
        # exp
        flops += prod(self.A_cum)
        #Mul YOFF
        flops += prod(self.C_b)
        #Add yoff+ydiag
        flops += prod(self.Y_diag)
        #Add y=y+D
        flops += prod(self.x_pad)

        return {self.vector_dtype: flops}