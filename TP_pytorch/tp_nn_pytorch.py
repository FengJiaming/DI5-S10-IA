import torch
import torch.nn as nn #Les réseaux de neurones peuvent être construits avec le paquet torch.nn.
import torch.optim as optim # Un package contenant de nombreux algorithmes d'optimisation
from torch.utils.data import Dataset
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
        
    def forward(self, x):  #forward (entrée) : Il renvoie la sortie(output).
        x = self.relu(self.layer_1(x)) #en utilisant fonction d'avtivitation
        x = self.relu(self.layer_2(x))
        x = nn.functional.softmax(self.out_layer(x))  #couche d'activation
        return x
    
class iris_dataset(Dataset): #sous-classes de torch.utils.data.Dataset, pour lire les données
    def __init__(self,data,target):
        super(iris_dataset,self).__init__()
        self.data = data
        self.labels = target
    
    def __len__(self): # override?
        return len(self.labels)
        
    def __getitem__(self, idx):  # Construire les tenseur de data et label
        data = self.data[idx,:]
        data = (torch.tensor(data,dtype=torch.float)).unsqueeze(0)
        label = self.labels[idx]
        label = torch.tensor(label,dtype= torch.long)
        return data,label
        
    
neural_network = model(30,2) # création du modèle avec 30 input_channels et 2 output_channels

# nn.Parameter : Une sorte de tenseur, qui est automatiquement enregistré en tant que paramètre lorsqu'il est attribué en tant qu'attribut à un module
# lr: learning rate = 0.1, ici c'est le pas du gradient descent.
#  x+=v,  v=−dx∗lr+v∗momemtum
#  nesterov:  d(x + mu * v)
opt = optim.SGD(params = neural_network.parameters(), lr = 0.1,momentum = 0.9, nesterov=True) #Créez un optim et spécifiez les paramètres correspondants
criterion = torch.nn.CrossEntropyLoss() # Fonction de perte(loss fonction), intégration de nn.logSoftmax () et nn.NLLLoss ()
#utiliser sklearn - datasets
dataset_iris = datasets.load_breast_cancer()  # Extraire des données, breast_cancer!!!
labels = dataset_iris.target # 0, 1 ['malignant' 'benign']
data = dataset_iris.data
data = (data-data.min())/(data.max()-data.min())
train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2)

dataset_iris = iris_dataset(train_data,train_labels)#torch dataset lire les données
#batch_size: Combien d'échantillons sont chargés dans chaque lot (par défaut: 1).
dataloader_train = torch.utils.data.DataLoader(dataset_iris,batch_size = 100)#DataLoader: Un type de données qui détermine comment les données sont entrées dans le réseau


y = []
for epoch in range(1000):
    i = 0
    loss_epoch = 0
    for data,label in dataloader_train:
        neural_network.zero_grad() #Initialiser le gradient de paramètre du modèle à 0
        output = neural_network(data) #Obtenir la sortie
        loss = criterion(output.squeeze(1),label) # calculer la perte entre vrais donnees et prédiction donnees
        loss_epoch = loss_epoch+loss.data.tolist() # sum de tous les perte
        loss.backward() # Gradient de rétropropagation
        opt.step() # Mettre à jour les paramètres
        i+=1
    print('Mean loss:', loss_epoch/i)  #Perte moyenne
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

