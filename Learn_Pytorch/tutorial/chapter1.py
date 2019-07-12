from __future__ import print_function
import torch

x1=torch.empty(5, 3)
print(x1)

x2=torch.rand(5, 3)
print(x2)

x3=torch.zeros(5, 3, dtype=torch.long)
print(x3)

x4=torch.tensor([5.5, 3])
print(x4)

# datatype override
x4=x4.new_ones(5, 3, dtype=torch.double)
print(x4)
x4=torch.randn_like(x4, dtype=torch.float)
print(x4)
print(x4.size())

# add tensor
y1=torch.rand(5, 3)
# method 1
print(x4+y1)
# method 2
print(torch.add(x4, y1))
# method 3
result=torch.empty(5, 3)
torch.add(x4, y1, out=result)
print(result)
# method 4 ('_' implies override)
y1.add_(x4)
print(y1)

print(x4[:,1])

# change the size or shape of the tensor
x5=torch.randn(4, 4)
y2=x5.view(16)
# -1 will be inferred from other dimensions
z1=x5.view(-1, 8)
print(x5.size(), y2.size(), z1.size())

x6=torch.randn(1)
print(x6)
print(x6.item())