import torch
import torch.nn as nn
from collections.abc import Iterable

class SGD_Optimizer():
    def __init__(self, params:list[torch.Tensor], lr:float):
        super(SGD_Optimizer, self).__init__()

        self.lr:float = lr
        self.num_states=len(params)

    # 定义 SGD 优化器
    def step(self,params:list[torch.Tensor])->list[torch.Tensor]:
        #检查参数和优化器状态的数量是否一致
        if len(params) != self.num_states:
            raise ValueError("Error: Number of parameters must match origin!")

        #实现公式 param=param-lr * grad
        lr = self.lr
        for i in range(self.num_states):
            #若参数不需要更新，直接跳过
            if params[i].grad is None:
                continue
            #获取梯度
            grad:torch.Tensor=params[i].grad
            update = -1.0 * lr * grad
            params[i]=params[i] + update

        return params


class Adagrad_Optimizer():
    def __init__(self, params:list[torch.Tensor], lr:float, epsilon:float=1e-8):
        super(Adagrad_Optimizer, self).__init__()

        self.lr:float = lr
        self.epsilon:float = epsilon
        self.velocities:list[torch.Tensor] = [torch.zeros_like(param) for param in params]

        self.num_states=len(self.velocities)

    # 定义 Adagrad 优化器
    def step(self,params:list[torch.Tensor])->list[torch.Tensor]:
        #检查参数和优化器状态的数量是否一致
        if len(params) != self.num_states:
            raise ValueError("Error: Number of parameters must match origin!")

        #实现Adagrad公式
        lr = self.lr
        epsilon = self.epsilon

        for i in range(self.num_states):
            #若参数不需要更新，直接跳过
            if params[i].grad is None:
                continue
            #获取梯度
            grad:torch.Tensor=params[i].grad

            #获取累计平方梯度
            r = self.velocities[i]
            #更新累计平方梯度
            r = r + torch.square(grad)
            #计算更新
            update = -1.0 * lr * grad / (torch.sqrt(r) + epsilon)
            #更新参数
            params[i] = params[i] + update

        return params



class RMSprop_Optimizer():
    def __init__(self, params:list[torch.Tensor], lr:float, beta:float=0.9, epsilon:float=1e-8):
        super(RMSprop_Optimizer, self).__init__()

        self.lr:float = lr
        self.epsilon:float = epsilon
        self.beta:float = beta
        self.velocities:list[torch.Tensor] = [torch.zeros_like(param) for param in params]

        self.num_states=len(self.velocities)

    # 定义 RMSprop 优化器
    def step(self,params:list[torch.Tensor])->list[torch.Tensor]:
        #检查参数和优化器状态的数量是否一致
        if len(params) != self.num_states:
            raise ValueError("Error: Number of parameters must match origin!")

        #实现RMSprop公式
        lr = self.lr
        epsilon = self.epsilon
        beta = self.beta

        for i in range(self.num_states):
            #若参数不需要更新，直接跳过
            if params[i].grad is None:
                continue
            #获取梯度
            grad:torch.Tensor=params[i].grad

            #获取累计平方梯度
            r = self.velocities[i]
            #更新累计平方梯度
            r = beta * r + ( 1.0 - beta ) * torch.square(grad)
            #计算更新
            update = -1.0 * lr * grad / (torch.sqrt(r + epsilon))
            params[i] = params[i] + update

        return params



class Adam_Optimizer():
    def __init__(self, params:list[torch.Tensor], lr:float, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        super(Adam_Optimizer, self).__init__()

        self.lr:float = lr
        self.beta1:float = beta1
        self.beta2:float = beta2
        self.epsilon:float = epsilon
        self.momentums:list[torch.Tensor] = [torch.zeros_like(param) for param in params]
        self.velocities:list[torch.Tensor] = [torch.zeros_like(param) for param in params]

        self.num_states=len(self.momentums)

    # 定义 Adam 优化器
    def step(self,params:list[torch.Tensor])->list[torch.Tensor]:
        #检查参数和优化器状态的数量是否一致
        if len(params) != self.num_states:
            raise ValueError("Error: Number of parameters must match origin!")

        #实现adam公式
        beta1 = self.beta1
        beta2 = self.beta2
        lr = self.lr
        epsilon = self.epsilon

        for i in range(self.num_states):
            #若参数不需要更新，直接跳过
            if params[i].grad is None:
                continue
            #获取梯度
            grad:torch.Tensor=params[i].grad

            #获取动量和速度
            m = self.momentums[i]
            v = self.velocities[i]

            #一阶矩估计
            m = beta1 * m + (1.0 - beta1) * grad
            #二阶矩估计
            v = beta2 * v + (1.0 - beta2) * torch.square(grad)
            #更新momentums和velocities
            self.momentums[i] = m
            self.velocities[i] = v
            #修正一阶矩偏差
            m_hat = m / (1.0 - beta1)
            #修正一阶矩偏差
            v_hat = v / (1.0 - beta2)
            #计算更新
            update = -1.0 * lr * m_hat / (torch.sqrt(v_hat) + epsilon)
            #更新参数
            params[i] = params[i] + update

        return params
