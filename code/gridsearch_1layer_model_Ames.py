import torch
import pandas as pd
import copy
import itertools
from pytorch_classes_and_functions import AmesDataset, DNN, do_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#store parameters and auc in lists for each hyperparameter combination
parameter_list = []
auc_list = []
best_auc = 0

#import data
df_train = pd.read_csv('../data/train_set_Ames.csv')
df_val = pd.read_csv('../data/val_set_Ames.csv')

#store parameter options in lists
radii = [1,2]
fp_bits = [2048]
units_per_layer = [512,1024,2048]
batch_size = [16,32,64]
l2_regularization = [0, 0.00001, 0.001]
dropout = [0,0.2,0.5]
learning_rate = [0.0001, 0.00033, 0.001]

for radius in radii:
    for bits in fp_bits: 
        #get train and val set with correct features
        Ames_train = AmesDataset(df_train,bits,radius)
        Ames_val = AmesDataset(df_val,bits,radius)

        for parameters in itertools.product(units_per_layer,batch_size,l2_regularization,dropout,learning_rate):
            upl = parameters[0]
            bs = parameters[1]
            l2 = parameters[2]
            drop = parameters[3]
            lr = parameters[4]
                                 
            #train model with hyperparameter combination, 1 hidden layers, 10 epochs max
            model = DNN(bits,upl,1,drop)
            model.to(device)
            _, auc, best_model_parameters = do_training(model,device,Ames_train,Ames_val,bs,lr,l2,10)
            
            if auc > best_auc:
                best_auc = auc
                best_model = copy.deepcopy(best_model_parameters)
            
            auc_list.append(auc)
            parameter_list.append('{}|{}|{}|{}|{}|{}|{}'.format(radius,bits,upl,bs,l2,drop,lr))
    
    
#create df with scrores of gridsearch
df_scores = pd.DataFrame(data={'parameters':parameter_list,'AUC':auc_list})
df_scores.to_csv('../results/scores_gridsearch_ffn_1layer_2048_input_Ames.csv',index=False)

#save best model
torch.save(best_model.state_dict(),'../pytorch_models/ffn_model_1layer_2048_input_state.pth') 
    