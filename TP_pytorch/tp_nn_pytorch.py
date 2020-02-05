import torch
import torch.nn as nn #Les réseaux de neurones peuvent être construits avec le paquet torch.nn.
import torch.optim as optim # Un package contenant de nombreux algorithmes d'optimisation
from sklearn.metrics import accuracy_score, precision_score
from torch.utils.data import Dataset # Présentation de l'ensemble de données
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class model(nn.Module):  #Définir un réseau neuronal avec des paramètres entraînables
    def __init__(self,input_channels,output_channels):
        super(model,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        #une opération affine: y = Wx + b
        self.layer_1 = nn.Linear(self.input_channels, 16)   # layer_1:  input channel,  16 output channels
        self.layer_2 = nn.Linear(16,8)  # layer_2: 16 input channel, 8 output channels
        self.out_layer = nn.Linear(8,self.output_channels,bias=False) # out_layer:  8 input channels, output_channels  .  bias:  non bias
        self.relu = nn.ReLU()
        
    def forward(self, x): # # propagation en avant
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = nn.functional.softmax(self.out_layer(x)) # softmax
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
        
    
neural_network = model(30,2) # Créer un modèle
opt = optim.SGD(params = neural_network.parameters(), lr = 0.1,momentum = 0.9, nesterov=True)
criterion = torch.nn.CrossEntropyLoss()

dataset_iris = datasets.load_breast_cancer() 
labels = dataset_iris.target
data = dataset_iris.data
data = (data-data.min())/(data.max()-data.min())

train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2) #train+validation:test=0.6+0.2:0.2

dataset_iris = iris_dataset(train_data,train_labels)#torch dataset lire les données

train_data, val_data, train_labels, val_labels = train_test_split(dataset_iris.data, dataset_iris.labels,
                                                                  test_size=0.25)# train:validation=0.6:0.2
dataset_test = iris_dataset(test_data, test_labels)  # dataset pour test
dataset_train = iris_dataset(train_data,train_labels) # dataset pour train
dataset_val = iris_dataset(val_data, val_labels)#dataset pour valitaion

#batch_size: Combien d'échantillons sont chargés dans chaque lot (par défaut: 1).
dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size = 100)#DataLoader: Un type de données qui détermine comment les données sont entrées dans le réseau
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=100)
dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=100)

mean_loss = []
accuracy = []
precision = []
accuracy_best = 0.0

for epoch in range(1000):
    i = 0
    loss_epoch = 0
    preds = []
    for data,label in dataloader_train:
        neural_network.zero_grad() #Initialiser le gradient de paramètre du modèle à 0
        output = neural_network(data) #Obtenir la sortie
        loss = criterion(output.squeeze(1),label) # calculer la perte entre vrais donnees et prédiction donnees
        loss_epoch = loss_epoch+loss.data.tolist() # sum de tous les perte
        loss.backward() # Gradient de rétropropagation
        opt.step() # Mettre à jour les paramètres
        i+=1

    for data, label in dataloader_val:
        output = neural_network(data)
        _, pred = torch.max(output.squeeze(1), dim=1)
        preds.append(pred)

    preds_tensor = preds[0]
    for j in range(len(preds)-1):
        preds_tensor = torch.cat((preds_tensor,preds[j+1]),0)

    if accuracy_best<accuracy_score(preds_tensor, val_labels):
        accuracy_best=accuracy_score(preds_tensor, val_labels)
        best_epoch=epoch

    print('Accuracy score: ', accuracy_score(preds_tensor, val_labels))
    print('Precision score: ', precision_score(preds_tensor, val_labels))
    print('Mean loss:', loss_epoch/i)
    accuracy.append(accuracy_score(preds_tensor, val_labels))
    precision.append(precision_score(preds_tensor, val_labels))
    mean_loss.append(loss_epoch/i)

#Visualizer
plt.figure()
x = range(0,1000)
plt.plot(x,mean_loss, label='mean loss')
plt.plot(x, accuracy, 'r', label='accuracy')
plt.plot(x, precision, 'b', label='precision')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

preds = []
for data, label in dataloader_test: #Test de modele avec de nouvelles données(dataset_test)
    output = neural_network(data)
    _, pred= torch.max(output.squeeze(1), dim=1)
    preds.append(pred)
preds_tensor = preds[0]
j=0
for j in range(len(preds)-1):
    preds_tensor = torch.cat((preds_tensor,preds[j+1]),0)

print('Accuracy test score: ', accuracy_score(preds_tensor, test_labels))
print('Precision test score: ', precision_score(preds_tensor, test_labels))

print('Best accuracy: ', accuracy_best)
print('Best epoch: ', best_epoch)



"""
1.Visualizer l'évolution de la loss, tracez la courbe de perte.
2.Calculer la précision du modèle sur la base d'apprentissage.    
3.Créer une base de validation et une base de test
4.Choisir le modèle qui donne le meilleur résultat sur la base de validation et retester ce modèle sur la base de test.

"""

