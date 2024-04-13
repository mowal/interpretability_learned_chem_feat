import pandas as pd
from rdkit.Chem import PandasTools
import torch
from pytorch_classes_and_functions import AmesDataset, DNN
from captum.attr import IntegratedGradients
from local_evaluation_functions import map_attr_to_atoms, attribution_auc
from helper_functions import string_to_int_list, dict_to_string, string_to_dict

#do_evaluation = False #set to true to get attribution AUC values, SDF file with alert atoms (i.e. ground truth) required
do_evaluation=True

df_val = PandasTools.LoadSDF('../data/val_set_Ames.sdf', idName='internalID', molColName='Molecule')
df_val.set_index('internalID', inplace=True)
df_val = df_val.astype({'Activity':float})

#get train and set with correct features
bits=2048
radius=1
n_hidden=512

#import model
device = torch.device('cpu')
model = DNN(bits,n_hidden, radius, 0)
model.load_state_dict(torch.load('../pytorch_models/ffn_model_1layer_2048_input_state.pth', map_location=device))
model.eval()

#create Ames datasets for torch
Ames_val = AmesDataset(df_val,activity_col='Activity',smiles_col='standardised_smiles',fingerprint_size=bits,fp_radius=radius)

#get attributions for val
baseline = torch.zeros(bits).reshape(1,bits)
ig = IntegratedGradients(model.forward)
attributions_val = ig.attribute(inputs=Ames_val.x, baselines=baseline)
attributions_val_np = attributions_val.detach().numpy()

#from bit attributions to atom attributions
attributions_atoms_val = []
for i,(idx,row) in enumerate(df_val.iterrows()):
    compound_dict = map_attr_to_atoms(idx,df_val,'Molecule',attributions_val_np[i,:].flatten(),
                                      radius,bits,ismol=True)
    attributions_atoms_val.append(compound_dict)

#store atom attributions in df
attributions_atoms_val_strings = []
for atom_attr in attributions_atoms_val:
    attributions_atoms_val_strings.append(dict_to_string(atom_attr,'|'))
    
df_attributions = pd.DataFrame(data={'atom_attributions':attributions_atoms_val_strings},index=df_val.index)
df_attributions.to_csv('../results/atom_attributions_val_iginput.csv')

if do_evaluation:
    #this is a dummy file containing alerts for just two examples (aromatic nitro) as Derek Nexus alerts are proprietary
    alerts_file = '../data/dummy_alert_atoms.sdf'
    df_alerts = PandasTools.LoadSDF(alerts_file, idName='internalID',molColName='Molecule')
    df_alerts.set_index('internalID', inplace=True)
                   
    #get attribution AUC for provided examples
    auc_list = []
    attr_list = []
    true_atoms_list = []
    
    for idx in df_alerts.index:
        #get atom attributions as dict
        attr_str = df_attributions.loc[idx,'atom_attributions']
        attr_dict = string_to_dict(attr_str,'|',int,float)
        attr_list.append(attr_str)
        #get true atoms
        true_atoms_str = df_alerts.loc[idx,'AlertAtoms']
        true_atoms = string_to_int_list(true_atoms_str,'|')
        true_atoms_list.append(true_atoms_str)
        #calculate attribution AUC
        auc_list.append(attribution_auc(true_atoms,attr_dict))
        
    df_evaluation = pd.DataFrame(index=df_alerts.index,data={'atom_attributions':attr_list,'AlertAtoms':true_atoms_list,'AttributionAUC':auc_list})
    df_evaluation.to_csv('../results/attribution_aucs_iginput.csv')
