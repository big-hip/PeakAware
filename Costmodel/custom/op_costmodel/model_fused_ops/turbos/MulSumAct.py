from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from math import prod

# MulSumAct (conv_state, self.conv1d.weight)
@op_manager.register("MulSumAct")
class MulSumAct(BaseOp):
	def __init__(self, op, hardware):
		'''
		input:
			(conv_state, self.conv1d.weight)

		output:
			xBC
		'''
		super().__init__(op, hardware)
		self.op = op
		self.op_type = "vector"
		self.instance = op.instance

		# x_flat, B_flat, ssm_state, dA, C, D, x = self.inputs_shape
		self.conv_state, self.conv1d_weight = self.inputs_shape
		self.vector_dtype = op.instance.vector_dtype

	@property
	def vec_flops_dict(self):
		flops = 0
		# flops+=1
		# Mul
		flops += prod(self.conv_state)
		# Sum
		flops += self.conv_state[-1]
		# Silu
		flops += prod(self.conv_state[:-1])

		return {str(self.vector_dtype): flops}