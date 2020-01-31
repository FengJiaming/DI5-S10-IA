import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(model,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.layer_1 = nn.Linear(self.input_channels, 16)
        self.layer_2 = nn.Linear(16,8)
        self.out_layer = nn.Linear(8,self.output_channels,bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = nn.functional.softmax(self.out_layer(x))
        return x
    
class iris_dataset(Dataset):
    def __init__(self,data,target):
        super(iris_dataset,self).__init__()
        self.data = data
        self.labels = target
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        data = self.data[idx,:]
        data = (torch.tensor(data,dtype=torch.float)).unsqueeze(0)
        label = self.labels[idx]
        label = torch.tensor(label,dtype= torch.long)
        return data,label
        
    
neural_network = model(30,2)
opt = optim.SGD(params = neural_network.parameters(), lr = 0.1,momentum = 0.9, nesterov=True)
criterion = torch.nn.CrossEntropyLoss()

dataset_iris = datasets.load_breast_cancer() 
labels = dataset_iris.target
data = dataset_iris.data
data = (data-data.min())/(data.max()-data.min())
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2)

dataset_iris = iris_dataset(train_data,train_labels)
dataloader_train = torch.utils.data.DataLoader(dataset_iris,batch_size = 100)

y = []
for epoch in range(1000):
    i = 0
    loss_epoch = 0
    for data,label in dataloader_train:
        neural_network.zero_grad()
        output = neural_network(data)
        loss = criterion(output.squeeze(1),label)
        loss_epoch = loss_epoch+loss.data.tolist()
        loss.backward()
        opt.step()
        i+=1
    print('Mean loss:', loss_epoch/i)
    y.append(loss_epoch/i)

plt.figure()
x = range(0,1000)
plt.plot(x,y)
plt.show()
"""
1.Visualizer l'évolution de la loss, tracez la courbe de perte.
2.Calculer la précision du modèle sur la base d'apprentissage.    
3.Créer une base de validation et une base de test
4.Choisir le modèle qui donne le meilleur résultat sur la base de validation et retester ce modèle sur la base de test.

"""

