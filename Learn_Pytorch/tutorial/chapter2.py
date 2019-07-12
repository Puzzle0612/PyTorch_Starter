import torch

a1=torch.randn(2, 2)
a1=((a1*3)/(a1-1))
print(a1.requires_grad)
a1.requires_grad_(True)
print(a1.requires_grad)
b1=(a1*a1).sum()
print(b1.grad_fn)

# auto_grad with scalar result
x1=torch.ones(2, 2, requires_grad=True)
print(x1)

y1=x1+2
print(y1)
print(y1.grad_fn)

z1=y1*y1*3
out=z1.mean()
print(z1, out)

out.backward()
print(x1.grad)

# auto_grad with vector result
x2=torch.randn(3, requires_grad=True)
y2=x2*2
while y2.data.norm()<1000:
    y2=y2*2
print(y2)

v1=torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y2.backward(v1)
print(x2.grad)


# wrapped with no_grad
print(x2.requires_grad)
print((x2**2).requires_grad)

with torch.no_grad():
    print((x2**2).requires_grad)



