from math import prod
from atencost.backend.analytical_model.op_costmodel.element_op import Element4Prediction
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager


@op_manager.register("RmsNormQuant")
class RmsNormQuantPrediction(Element4Prediction):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.init_vector_instruction()

    def init_vector_instruction(self):
        if (len(self.inputs_shape[0]) == 2):
            m, n = self.inputs_shape[0]
        else:
            m = 1
            for dim in self.inputs_shape[0][:-1]:
                m *= dim
            n = self.inputs_shape[0][-1]

        dtype_name = str(self.inputs_dtype[0]).split(".")[-1]
        if dtype_name == 'bfloat16':
            dtype_name = 'float16'
        self.vec_instruction = [
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VDIV", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VABS", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VLN", dtype_name, m, n],
            ["VMAX", dtype_name, m, n],
            ["VSUB", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
        ]

@op_manager.register("Quant")
class QuantPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.init_vector_instruction()

    def init_vector_instruction(self):
        if (len(self.inputs_shape[0]) == 2):
            m, n = self.inputs_shape[0]
        else:
            m = 1
            for dim in self.inputs_shape[0][:-1]:
                m *= dim
            n = self.inputs_shape[0][-1]

        dtype_name = str(self.inputs_dtype[0]).split(".")[-1]
        if dtype_name == 'bfloat16':
            dtype_name = 'float16'
        self.vec_instruction = [
            ["VABS", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VLN", dtype_name, m, n],
            ["VMAX", dtype_name, m, n],
            ["VSUB", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
        ]

@op_manager.register("AddRmsNormQuant")
class AddRmsNormQuantPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.init_vector_instruction()

    def init_vector_instruction(self):
        if (len(self.inputs_shape[0]) == 2):
            m, n = self.inputs_shape[0]
        else:
            m = 1
            for dim in self.inputs_shape[0][:-1]:
                m *= dim
            n = self.inputs_shape[0][-1]

        dtype_name = str(self.inputs_dtype[0]).split(".")[-1]
        if dtype_name == 'bfloat16':
            dtype_name = 'float16'
        self.vec_instruction = [
            ["VADD", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VDIV", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VABS", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VLN", dtype_name, m, n],
            ["VMAX", dtype_name, m, n],
            ["VSUB", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
        ]


@op_manager.register("RopeQuant")
class RopeQuantPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.init_vector_instruction()

    def init_vector_instruction(self):
        if (len(self.inputs_shape[0]) == 2):
            m, n = self.inputs_shape[0]
        else:
            m = 1
            for dim in self.inputs_shape[0][:-1]:
                m *= dim
            n = self.inputs_shape[0][-1]

        dtype_name = str(self.inputs_dtype[0]).split(".")[-1]
        if dtype_name == 'bfloat16':
            dtype_name = 'float16'
        self.vec_instruction = [
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VABS", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VLN", dtype_name, m, n],
            ["VMAX", dtype_name, m, n],
            ["VSUB", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n / 2],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VMUL", dtype_name, m, n],
        ]
