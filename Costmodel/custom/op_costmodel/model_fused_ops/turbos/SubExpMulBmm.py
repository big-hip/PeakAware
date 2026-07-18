from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from math import prod

# SubExpMulBmm (A_cum, B_r, x_b)
@op_manager.register("SubExpMulBmm")
class SubExpMulBmm(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
			A_cum
			B_r
			x_b

        output:
            states

        timeline (ignore reshape)
            decay = torch.exp(A_cum[:, :, :, -1:] - A_cum).to(torch.bfloat16)
            decay_states_us = decay.unsqueeze(-1)
            Bd_r = (B_r.unsqueeze(2)) * (decay_states_us.view(b,g,-1,c,l,1))
            states = torch.bmm(Bd_b,x_b).transpose(1,2).view(b,-1,c,x_r.shape[-1],l).permute(0,2,1,3,4).contiguous()
        '''
        super().__init__(op, hardware)
        self.op = op
        self.op_type = "mix"
        self.instance = op.instance

        # x_flat, B_flat, ssm_state, dA, C, D, x = self.inputs_shape
        self.A_cum, self.B_r , self.x = self.inputs_shape
        self.bmm_dtype =  self.inputs_dtype[-1]
        self.vector_dtype = op.instance.vector_dtype


    @property
    def cube_flops_dict(self):
        flops = 2 * prod(self.x) * self.x[-1]
        return {self.bmm_dtype: flops}

    @property
    def vec_flops_dict(self):
        flops = 0
        # flops+=1
        # Sub, exp
        flops += 2 * prod(self.A_cum)

        # Mul Bd_r line 337
        flops += prod(self.B_r[:-1])*self.A_cum[-1]

        return {str(self.vector_dtype): flops}