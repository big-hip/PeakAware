from math import prod

import torch

from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager

from atencost.backend.analytical_model.utils.datatype import get_dtype_size


@op_manager.register("AdamOptimizerStep")
class AdamOptimizerStepPrediction(BaseOp):
    def __init__(self, op, hardware):
        '''
        input:
            model_parameter_numel(int): 模型参数量
            dtype_str(str): 转成字符串的torch.dtype

        优化器内部的计算应该都是
        '''
        super().__init__(op, hardware)
        self.model_parameter_numel, dtype_str = op.inputs
        self.dtype = eval(dtype_str)
        self.dtype_val = get_dtype_size(self.dtype)

    @property
    def vec_flops_dict(self):
        flops = 14 * self.model_parameter_numel
        return {self.dtype: flops}

    @property
    def memory_size(self):
        size = 7 * self.model_parameter_numel * self.dtype_val
        return size

@op_manager.register("ApplyAdamW")
class NpuAdamOptimizerStepPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input:
            model_parameter_numel(int): 模型参数量
            dtype_str(str): 转成字符串的torch.dtype

        优化器内部的计算应该都是
        """
        super().__init__(op, hardware)
        self.model_parameter_numel = 0
        self.model_parameter_numel += prod(self.inputs_shape[0])
        self.dtype = self.inputs_dtype[0]
        self.dtype_val = get_dtype_size(self.dtype)

    @property
    def vec_flops_dict(self):
        flops = 14 * self.model_parameter_numel
        return {self.dtype: flops}

    @property
    def memory_size(self):
        size = 7 * self.model_parameter_numel * self.dtype_val
        return size
