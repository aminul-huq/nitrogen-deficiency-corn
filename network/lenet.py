import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*53*53, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        #print(out.shape)
        out = F.max_pool2d(out, 2)
        #print(out.shape)
        out = F.relu(self.conv2(out))
        #print(out.shape)
        out = F.max_pool2d(out, 2)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def test():
    net = LeNet(num_classes=10)
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

test()