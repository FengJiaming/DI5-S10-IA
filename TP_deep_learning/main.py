import numpy as np
import torch
from torch import nn
from torch import autograd
import math
from torch.utils.data import DataLoader
from tiny_imagenet_dataset import TinyImageNetDatasetTrain
from tiny_imagenet_dataset import TinyImageNetDatasetTest
import cv2
import sys

import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torchvision

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5)

        self.fc1 = nn.Linear(16*13*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)
   

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*13*13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net=Net()
PATH = './cifar_net.pth'
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0

img_dir = './data'
datasetTrain = TinyImageNetDatasetTrain(img_dir)
loaderTrain = DataLoader(datasetTrain, shuffle = True, batch_size = 1)

datasetTest = TinyImageNetDatasetTest(img_dir)
loaderTest = DataLoader(datasetTest, shuffle = True, batch_size = 1)

for epoch in range(2):
    for i, datum in enumerate(loaderTrain):
        img, idclass = datum
        #print(img)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(img)
        loss = criterion(outputs, idclass)
        loss.backward()
        optimizer.step()

#        print(statistics)
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    

torch.save(net.state_dict(), PATH)

#dataiter = iter(loaderTest)
#images, labels = dataiter.next()
#
## print images
#imshow(torchvision.utils.make_grid(images))
#print(labels)
#
#net.load_state_dict(torch.load(PATH))
#
#outputs = net(images)
#
#_, predicted = torch.max(outputs, 1)
#
#print(predicted)
