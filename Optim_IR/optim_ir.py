from .optim_algorithm import (
    SGD_Optimizer,
    Adagrad_Optimizer,
    RMSprop_Optimizer,
    Adam_Optimizer,)

from IR_transform import (
    torch_compile_capture,
    aten_compile_capture,
    core_aten_compile_capture,
    prims_compile_capture,
    prims_compile_capture)

import torch
from typing import Callable

def optim_capture(opt_type:str,
                     params:list[torch.Tensor],
                     lr:float=0.001,
                     beta1:float=0.9,
                     beta2:float=0.999,
                     epsilon:float=1e-8,
                     need_compile:bool=True)-> Callable:
    #检查优化器类型
    if opt_type not in ["sgd", "adagrad", "rmsprop", "adam"]:
        raise ValueError("Error: Invalid optimizer type, please choose from 'sgd', 'adagrad', 'rmsprop', 'adam'")

    if opt_type == "sgd":
        opt=SGD_Optimizer(params,lr)
    elif opt_type == "adagrad":
        opt=Adagrad_Optimizer(params,lr,epsilon)
    elif opt_type == "rmsprop":
        opt=RMSprop_Optimizer(params,lr,beta1,epsilon)
    elif opt_type == "adam":
        opt=Adam_Optimizer(params,lr,beta1,beta2,epsilon)
    #TODO:添加adamW优化器
    elif opt_type == "adamw":
        pass
        #adamW与adam的区别，在于对权重衰减的实现
        #opt=AdamW_Optimizer(params,lr=0.001)

    if need_compile:
        opt_compiled=aten_compile_capture(opt.step)
        return opt_compiled
    else:
        return opt.step
