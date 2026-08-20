from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager
from math import prod

# bmmmulbmm(C_b, B_b, L_b, x_b)
@op_manager.register("BmmMulBmm")
class BmmMulBmm(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
           C_b
           B_b
           L_b
           x_b

        output:
          Y_diag
        '''
        super().__init__(op, hardware)
        self.op = op
        self.op_type = "mix"
        self.instance = op.instance

        # x_flat, B_flat, ssm_state, dA, C, D, x = self.inputs_shape
        self.C_b, self.B_b , self.L_b, self.x_b = self.inputs_shape
        self.bmm_dtype =  self.inputs_dtype[-1]
        self.vector_dtype = op.instance.vector_dtype


    @property
    def cube_flops_dict(self):
        flops_bmm1 = 2 * prod(self.C_b) * self.B_b[-1]
        flops_bmm2 = 2 * prod(self.x_b) * self.x_b[-1]
        return {self.bmm_dtype: flops_bmm1 + flops_bmm2}

    @property
    def vec_flops_dict(self):

        flops = prod(self.L_b)

        return {str(self.vector_dtype): flops}