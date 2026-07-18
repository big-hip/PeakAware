import torch
import torch.nn as nn
#import torch.optim as optim
import Dist_IR
from Dist_IR.Optim_IR import *


# 定义一个简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        # @Dist_IR.PP_stage_split
        out = self.relu(out)
        # @Dist_IR.PP_stage_split
        out = self.fc2(out)
        return out

# 超参数
input_size = 10
hidden_size = 20
output_size = 1
learning_rate = 0.01
num_epochs = 100
batch_size = 100  #  新增 batch size

# 初始化模型、损失函数、优化器
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()

optimizer = Adagrad_Optimizer(list(model.parameters()), lr=learning_rate)
# class RMSprop_Optimizer():
# class Adagrad_Optimizer():
# class SGD_Optimizer():
# class Adam_Optimizer():

# 构造输入数据（带 batch 维度）
x = torch.randn(batch_size, input_size)  # shape: [B, 10]
y = torch.randn(batch_size, output_size) # shape: [B, 1]

# 捕获图
""" 用户参考 start"""
graph_capture = Dist_IR.FullGraphCapture(model,optimizer,x)
model,optimizer = graph_capture.compile()
""" 用户参考 end"""

model.train()
for epoch in range(1):
    outputs = model(x)
    loss = criterion(outputs, y)

    #optimizer.zero_grad()
    loss.backward()
    #optimizer.step()
    params = [param for param in model.parameters() if param.grad is not None]
    grads = [param.grad for param in params]
    optimizer(params, grads)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 调用 pass
"""用户参考 start"""
Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm,graph_capture.OPT_gm, batch_size)
"""用户参考 end"""
