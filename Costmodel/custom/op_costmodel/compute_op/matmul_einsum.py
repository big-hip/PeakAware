import re
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager

from atencost.backend.analytical_model.utils.datatype import get_dtype_size
from math import prod


@op_manager.register("MatmulEinsum")
class MatmulEinsumPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            equation (str): e.g. 'bi,ij->bj'
            x (Tensor): e.g. [b,i]
            y (Tensor): e.g. [i,j]

        output(Tensor): [b,j]
        """
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.equation = op.inputs[0]

    @property
    def cube_flops_dict(self):
        dtype1, dtype2 = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        inputs_part, output_part = self.equation.split("->")
        input_subscripts = inputs_part.split(",")

        all_dims = set(re.sub(r"[,\-\>]", "", self.equation))

        dim_value_map = {}
        for subscript, shape in zip(input_subscripts, self.inputs_shape):
            dims = [d for d in subscript if d.isalpha()]
            for dim_name, size in zip(dims, shape):
                if dim_name not in dim_value_map:
                    dim_value_map[dim_name] = size

        flops = 2 * prod([dim_value_map[d] for d in all_dims])
        return {self.compute_type: flops}

    @staticmethod
    def infer_shape(equation, shape1, shape2):
        input_part, output_part = equation.split("->")
        input_subscripts = input_part.split(",")
        output_subscript = output_part

        shape_list = [shape1, shape2]
        dim_sizes = {}

        for subscript, shape in zip(input_subscripts, shape_list):
            for i, char in enumerate(subscript):
                if char in dim_sizes:
                    assert dim_sizes[char] == shape[i], f"Dimension mismatch for subscript '{char}'"
                else:
                    dim_sizes[char] = shape[i]

        output_shape = [dim_sizes[char] for char in output_subscript]

        return output_shape
