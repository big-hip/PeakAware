import torch
import torch.nn as nn
#import torch.optim as optim
from Dist_IR.Optim_IR import (SGD_Optimizer,
    Adagrad_Optimizer,
    RMSprop_Optimizer,
    Adam_Optimizer)

import Dist_IR



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
num_epochs = 1
batch_size = 100  #  新增 batch size

# 初始化模型、损失函数、优化器
model = SimpleMLP(input_size, hidden_size, output_size)

criterion = nn.MSELoss()

# 构造输入数据（带 batch 维度）
x = torch.randn(batch_size, input_size)  # shape: [B, 10]
y = torch.randn(batch_size, output_size) # shape: [B, 1]

# 捕获前向、反向图
""" 用户参考 start"""
graph_capture = Dist_IR.GraphCapture(model,x)
model = graph_capture.compile()

# 定义优化器，并捕获优化器图
#optimizer = Adam_Optimizer(list(model.parameters()), lr=learning_rate)
optimizer = RMSprop_Optimizer(list(model.parameters()), lr=learning_rate)
optim_graph_capture = Dist_IR.OptimGraphCapture(optimizer)
optimizer=optim_graph_capture.compile()

""" 用户参考 end"""

model.train()
for epoch in range(num_epochs):
    outputs = model(x)
    loss = criterion(outputs, y)

    #optimizer.zero_grad()
    loss.backward()
    #optimizer.step()
    optimizer(list(model.parameters()))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 调用 pass
"""用户参考 start"""
Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm,batch_size)
"""用户参考 end"""
