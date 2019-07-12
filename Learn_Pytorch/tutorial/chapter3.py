import torch
import torch.nn as nn
import torch.nn.functional as F

# define a net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # kernel
        # 1 to 6 with 5*5 & 6 to 16 with 5*5
        self.conv1=nn.Conv2d(1, 6, 5)
        self.conv2=nn.Conv2d(6, 16, 5)
        # y=Wx+b
        # full-convolution 16*5*5 -> 120 -> 84 -> 10
        self.fc1=nn.Linear(16*5*5, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 10)

    def forward(self, x):
        # 32*32
        x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # 6*14*14
        x=F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 16*5*5
        x=x.view(-1, self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        # 120
        x=F.relu(self.fc2(x))
        # 84
        x=self.fc3(x)
        # 10
        return x

    # given 1*16*5*5, calculate 16*5*5
    def num_flat_features(self, x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features

net=Net()
print(net)

params=list(net.parameters())
print(len(params))
i=0
while i<len(params):
    print(params[i].size())
    i+=1

input=torch.randn(1, 1, 32, 32)
out=net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

output=net(input)
target=torch.randn(10)
target=target.view(1, -1)
criterion=nn.MSELoss()

loss=criterion(output, target)
print(loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

net.zero_grad()
print('con1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('con1.bias.grad before backward')
print(net.conv1.bias.grad)

learning_rate=0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)

#import torch.optim as optim

# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)
#
# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update

