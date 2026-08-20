from atencost.backend.perf_result import OpPerfResult
from atencost.frontend.utils.tensor_record import TensorRecord


class BaseModel:
    def __init__(self, op, hardware):
        self.op = op
        self.hardware = hardware

        # 提取所有TensorRecord类型的输入元素
        self.tensor_inputs = [
            inp for inp in getattr(op, "inputs", []) if isinstance(inp, TensorRecord)
        ]

        self.parameter_inputs = [
            str(inp) for inp in getattr(op, "inputs", []) if not isinstance(inp, TensorRecord)
        ]
        # 从提取的TensorRecord元素中获取形状和数据类型
        self.inputs_shape = [inp.local_shape for inp in self.tensor_inputs]
        self.inputs_dtype = [str(inp.dtype) for inp in self.tensor_inputs]

        # 对输出做同样的处理
        self.tensor_outputs = [
            out for out in getattr(op, "outputs", []) if isinstance(out, TensorRecord)
        ]

        self.outputs_shape = [out.local_shape for out in self.tensor_outputs]
        self.outputs_dtype = [str(out.dtype) for out in self.tensor_outputs]

        self.result = OpPerfResult(inputs_shape=self.inputs_shape,
                                       inputs_dtype=self.inputs_dtype,
                                       parameters_input=self.parameter_inputs,
                                       outputs_shape=self.outputs_shape,
                                       outputs_dtype=self.outputs_dtype,
                                       module_path=self.op.module_path,
                                       global_rank_list=self.op.instance.global_rank_list if hasattr(self.op.instance, 'global_rank_list') else [])

    def __call__(self) -> OpPerfResult:
        return self.result
