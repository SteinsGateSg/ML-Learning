"""
我们的模型类必须继承nn.module, 它是所有神经网络模型的基础类
两个必须的函数
1. __init__() 这个不用说
2. forward()连名字都不能变
3. 这个模型会自动构建计算图，backward自动算出来了，不用人工写

class torch.nn.Linear(in_features, out_features, bias = True)
in_features : size of each input sample
out_features : size of each output sample
bias : if set to False, the layer will not lear an additive bias Default : False

class nn.Linear有一个个魔法方法__call__()，能使这个类能像一个函数一样被调用

class torch.nn.MESloss(reduction)
size_average : 要不要求均值
reduce : 要不要降维
以上已经弃用
reduction 'sum', 'mean', 'none'
"""

import torch
import torch.nn as nn

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

class LinearModel(nn.Module):

    def __init__(self):
        super().init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinearModel()
criterion = nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()