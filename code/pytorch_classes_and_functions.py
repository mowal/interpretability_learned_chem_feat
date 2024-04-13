import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import copy

class AmesDataset(Dataset):
    """
    dataset object
    mols represented as SMILES
    name of activity and SMILES column need to be specified
    featurisation as binary Morgan FP (size and radius can be specified)
    """
    
    def __init__(self, dataframe, activity_col='Activity',smiles_col='SMILES', fingerprint_size=2048, fp_radius=2):
        #get tensors for x and y from the pandas df
        self.y = torch.tensor(dataframe[activity_col].values,dtype=torch.float)
        self.n_samples = dataframe.shape[0]
        
        #mol indexes not needed
        dataframe = dataframe.copy()
        dataframe.reset_index(drop=True, inplace=True)
        
        #get X as Morgan Fingerprint
        X = np.empty((self.n_samples,fingerprint_size))
        
        for i in range(self.n_samples):
            X[i,:] = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(dataframe.loc[i,smiles_col]), fp_radius, nBits=fingerprint_size)
            
        self.x = torch.from_numpy(X).float()
    
    def __getitem__(self, index):
        return(self.x[index],self.y[index])
    
    def __len__(self):
        return(self.n_samples)

class DNN(nn.Module):
    """
    simple feedforward DNN class where all hidden layers have same number of neurons and same dropout
    """
    def __init__(self, input_shape=2048, hidden_shape=2048, n_hidden=1, dropout=0):
        super(DNN, self).__init__()
        self.input_shape = input_shape 
        self.hidden_shape = hidden_shape
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.hiddens = nn.ModuleList([nn.Linear(self.input_shape,self.hidden_shape) if i==0 else nn.Linear(self.hidden_shape,self.hidden_shape) for i in range(self.n_hidden)])
        self.output = nn.Linear(self.hidden_shape,1)
        self.dropout = nn.Dropout(p=self.dropout)
        
    def forward(self,x):
        for module in self.hiddens: # loop through hidden layers
            x = F.relu(module(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.output(x)) #output layer
        return x
    
    def hidden_activations(self,x):
        """
        function that returns activations of all hidden neurons as list, one list element per hidden layer
        """
        activations_per_layer = []
        for module in self.hiddens:
            x = F.relu(module(x))
            activations_per_layer.append(x.detach().numpy())
            
        return activations_per_layer
    
def do_training(model, device, train_data, val_data, batch_size=32, lr=0.001, l2=0, epochs=10):
    """
    function that performs training, returns AUC of best model and best model
    
    model: Pytorch model instance of class DNN
    device: "cuda:0" or "cpu"
    train_data: instance of AmesDataset
    val_data: instance of AmesDataset
    batch_size: int
    lr: learning rate, float
    l2: size of L2 regularization of weights, float
    epochs: number of maximal trianing epochs, int
    
    returns: best epoch (int), best AUC (float), best model (instance of DNN)
    """
    trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    
    best_auc = 0
    
    for epoch in range(epochs):
        model.train() #model in train mode (dropout enabled)
        
        for i, data in enumerate(trainloader):
            features,labels = data
            features = features.to(device)
            labels = labels.to(device)
            #set gradients to zero
            optimizer.zero_grad()
            #forward + backward + optimise
            outputs = model.forward(features).view((-1))
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        
        #get auc on validation after each epoch
        model.eval() #model in evaluation mode (dropout disabled)
        val_x = val_data.x.to(device)
        val_preds = model.forward(val_x)
        auc = roc_auc_score(val_data.y,val_preds.cpu().detach().numpy())
        if auc>best_auc:
            best_auc=auc
            best_epoch = epoch
            best_model = copy.deepcopy(model) #copy model
        print('AUC on val set after epoch {} : {}'.format(epoch,auc))
    
    return(best_epoch,best_auc,best_model)
