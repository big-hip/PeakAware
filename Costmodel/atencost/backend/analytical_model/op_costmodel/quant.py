from numpy import dtype
from atencost.frontend.utils.tensor_record import CustomDType
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager

from atencost.backend.analytical_model.utils.datatype import get_dtype_size
from atencost.backend.analytical_model.utils.helper import broadcast_shapes, create_op_info
from math import prod


@op_manager.register("Quantization", "ImpQuant", "Dequantization")
class StaticQuantPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):
        if op.raw_name == "ActivationQuant":
            """
            1. Compute the max/min value in activation
            2. Load the activation scaler/zero-point
            3. Quantize the activation: clip(round(x/scale) + zp, min, max)
            """
            # Why result[self.inputs_dtype[index]] ?
            # It is possible that input tensors have different dtype, each dtype's quantization performance maybe not the same
            flops = 0
            result = {}
            for index, tensor in enumerate(self.input_tensors):
                total_elements = prod(shape)
                flops = total_elements * 6 # step1 + step2
                if self.inputs_dtype[index] not in result:
                    result[self.inputs_dtype[index]] = flops
                else:
                    result[self.inputs_dtype[index]] += flops
            return result
        else: # For ImpQuant and Dequant
            """
            Once we have the output WX with type INT8 from previous op, then we want to compute ACC
            TODO:: We need to check the value of z_W and z_X here
                - Can we record the info in OpRecord?
            1. If W and X are all symmetric quantization (zero-point=0)
                - WX is ACC
            2. If one of them is not symmetric
                - ACC = WX + z_W*X
                - ACC = WX + z_X*W : can be off-line stored
                    - load z_X * W
                - FLOPS = total_elemtents of WX
            3. If both of them are not symmetric
                - ACC = WX + z_W*X + z_X*W
            - Compute the max/min value in ACC
            - Load the scaler and zero-point of this layer (here we can record last quant op's scaler and zero-point)
            - Quantize: clip(round(sWsXACC)/sY + zY, min, max)
            """
            flop_acc = 0
            scale = 0
            result = {}
            if "X_unsym_W_unsym" in op.raw_name:
                scale = 2
            elif "X_sym_W_sym" in op.raw_name:
                scale = 0
            else:
                scale = 1
            if "Dequantization" in op.raw_name:
                scale += 2
            for output_tensor in self.tensor_outputs:
                cur_dtype = output_tensor.dtype
                shape = output_tensor.local_shape
                total_elements = scale * prod(shape)
                if cur_dtype not in result:
                    result[cur_dtype] = total_elements
                else:
                    result[cur_dtype] += total_elements
            return result
