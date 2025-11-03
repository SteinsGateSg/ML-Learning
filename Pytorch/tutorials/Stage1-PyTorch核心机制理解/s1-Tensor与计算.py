import torch

# 创建张量
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
y = torch.tensor([[2., 2.], [2., 2.]])

z = x * y + 3
out = z.mean()
print(out)                  # 输出标量
out.backward()              # 自动求导
print(x.grad)               # 输出梯度