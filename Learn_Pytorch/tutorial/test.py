from chapter4 import Net
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader=torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    net=Net()
    net.load_state_dict(torch.load('./tutorial/net_params_10epoch.pkl'))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images.to(device)
            labels.to(device)
            outputs = net(images)
            outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))