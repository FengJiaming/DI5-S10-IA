import numpy as np
import torch
from torch import nn
from torch import autograd
import math
from torch.utils.data import DataLoader
from tiny_imagenet_dataset import TinyImageNetDatasetTrain
from tiny_imagenet_dataset import TinyImageNetDatasetTest
from tiny_imagenet_dataset import TinyImageNetDatasetValidation
import cv2
import sys

import torch.optim as optim
import torch.nn.functional as F

#lib to show an image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

#lib to show confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5)
#        self.conv3 = nn.Conv2d(16, 16, 7)   #Excercie 2
#        self.fc1 = nn.Linear(16*3*3, 120)
        self.fc1 = nn.Linear(16*13*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)
   

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#        x = self.pool(F.relu(self.conv3(x)))    #Excercie 2
#        x = x.view(-1, 16*3*3)
        x = x.view(-1, 16 * 13 *13)
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

datasetValidation = TinyImageNetDatasetValidation(img_dir)
loaderValidation = DataLoader(datasetValidation, shuffle = True, batch_size = 1)

datasetTest = TinyImageNetDatasetTest(img_dir)
loaderTest = DataLoader(datasetTest, shuffle = True, batch_size = 1)



#Excercice 1 fonction de cout chaque epochs 
lossEpoch = []
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

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    # Excercice 3
    lossValidatoin = 0.0
    for i, datum in enumerate(loaderValidation):
        img, idclass = datum
        net.load_state_dict(torch.load(PATH))
        outputs = net(img)
        loss = criterion(outputs, idclass)
        lossValidatoin += loss.item() 
    lossEpoch.append(lossValidatoin / 1000) 
    print('Validation Epoch %d .loss: %.3f' % (epoch, lossEpoch[epoch]))
    torch.save(net.state_dict(), PATH)
    if lossEpoch[epoch] <lossEpoch[epoch-1]:   
        torch.save(net.state_dict(), PATH)

#Excercice 3 Ensemble de validation
predicts = []
y_true=[]
running_loss = 0.0
for i, datum in enumerate(loaderValidation):
    img, idclass = datum
    net.load_state_dict(torch.load(PATH))
    outputs = net(img)
    loss = criterion(outputs, idclass)
    running_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    predicts.append(predicted.int())
    y_true.append(idclass.int())
print('loss: %.3f' % (running_loss / 1000))

#Excercice 1 fonction de cout et confusion matrix (Ensemble de test)
predicts = []
y_true=[]
running_loss = 0.0
for i, datum in enumerate(loaderTest):
    img, idclass = datum
    net.load_state_dict(torch.load(PATH))
    outputs = net(img)
    loss = criterion(outputs, idclass)
    running_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    predicts.append(predicted.int())
    y_true.append(idclass.int())
print('loss: %.3f' % (running_loss / 1000))

labels=[]
for i in range(1,20):
    id_class = torch.tensor(i).type(torch.int64)
    labels.append(id_class)
    
C2= confusion_matrix(y_true, predicts, labels)
print(C2)


# Code pour afficher les resultats
dataiter = iter(loaderTest)
images, label = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print(label)
net.load_state_dict(torch.load(PATH))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print(predicted)

