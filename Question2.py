import torch
import torch.nn as nn

x = torch.tensor([[1.0], [1.0]])
y = torch.tensor([[0.0]])

w = torch.tensor([[1.0, 1.0],[-1.0, -1.0]], requires_grad=True)
b = torch.tensor([[-0.5],[1.5]],requires_grad=True)
q = torch.tensor([[1.0],[1.0]],requires_grad=True)
c = torch.tensor(-1.5,requires_grad=True)
h = torch.mm(w,x)+b
h_relu = h.clamp(min = 0)
yy = torch.mm(h_relu.t(),q)+c
y_s = torch.sigmoid(yy)
becloss = nn.BCELoss()
loss = becloss(y_s,y)
loss.backward()

print('loss is {0}'.format(loss))
print(w.grad)
print(q.grad)
print(b.grad)
print(c.grad)