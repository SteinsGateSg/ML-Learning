import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    loss = (y - y_pred) ** 2
    return loss

print('predict (before training)', 4, forward(4, w.item()))

epoch_list = []
loss_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y, w)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print('process:', epoch, l.item())
    epoch_list.append(epoch)
    loss_list.append(l.item())

print('predict (after training)', 4, forward(4, w))

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()