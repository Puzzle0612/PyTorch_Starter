import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    # initialize a net
    def __init__(self):
        super(Net, self).__init__()
        # set kernel for convolutional operation
        # 1 to 6 with 5*5 & 6 to 16 with 5*5
        self.conv1=nn.Conv2d(3, 6, 5)
        self.conv2=nn.Conv2d(6, 16, 5)
        # y=Wx+b
        # set linear for fully-connection
        # 16*5*5 -> 120 -> 84 -> 10
        self.fc1=nn.Linear(16*5*5, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 10)

    # define forwarding process
    def forward(self, x):
        # MaxPooling
        # 32*32
        x=F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # 6*14*14
        x=F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 16*5*5
        # Change dimensions
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

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net=Net();
    net.cuda()
    # define loos function
    criterion=nn.CrossEntropyLoss()
    # define optimizer(learning rate and momentum)
    optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # training ...
    for epoch in range(10):
        running_loss=0.0
        for i,data in enumerate(trainloader, 0):
            inputs, labels=data
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=net(inputs)
            outputs=outputs.to(device)
            loss=criterion(outputs, labels)
            loss.cuda()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%2000==1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss=0.0
    print('Finished Training')
    torch.save(net.state_dict(), 'net_params_2epoch.pkl')

