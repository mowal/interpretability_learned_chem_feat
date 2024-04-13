import pandas as pd
from rdkit.Chem import PandasTools
import torch
from pytorch_classes_and_functions import AmesDataset, DNN
from captum.attr import LayerIntegratedGradients
from local_evaluation_functions import attribution_auc, get_attribution_for_hidden_layer
from helper_functions import string_to_int_list,dict_to_string, convert_atom_weights_col, string_to_set, string_to_dict

do_evaluation = False #set to true to get attribution AUC values for tP compounds, SDF file with alert atoms (i.e. ground truth) required

df_val = PandasTools.LoadSDF('../data/val_set_Ames.sdf', idName='internalID', molColName='Molecule')
df_val.set_index('internalID', inplace=True)
df_val = df_val.astype({'Activity':float})

df_substructures = pd.read_csv('../results_ames_test/substructures.csv')
df_substructures = convert_atom_weights_col(df_substructures)

#turn columns in df_substructures back to set type
supp_comp_sets = []
dir_par_sets = []
dir_chi_sets = []
for i,row in df_substructures.iterrows():
    supp_comp_sets.append(string_to_set(row['supporting_compounds']))
    dir_par_sets.append(string_to_set(row['direct_parents']))
    dir_chi_sets.append(string_to_set(row['direct_childs']))   
df_substructures['supporting_compounds'] = supp_comp_sets
df_substructures['direct_parents'] = dir_par_sets
df_substructures['direct_childs'] = dir_chi_sets

#import model
device = torch.device('cpu')
model = DNN(2048,512, 1, 0)
model.load_state_dict(torch.load('../pytorch_models/ffn_model_1layer_2048_input_state.pth', map_location=device))
model.eval()

#get train and set with correct features
bits=2048
radius=1

#create Ames datasets for torch
Ames_val = AmesDataset(df_val,activity_col='Activity',smiles_col='standardised_smiles',fingerprint_size=bits,fp_radius=radius)

#get layer integrated gradients
baseline = torch.zeros(bits).reshape(1,bits)
lig = LayerIntegratedGradients(model.forward,model.hiddens[0],multiply_by_inputs=True)
attributions_val = lig.attribute(inputs=Ames_val.x,baselines=baseline).detach().numpy()

#whether atom weighting scheme is applied ot obtain atom attributions
weighting_scheme = True
attributions_atoms_val = []
attribution_atoms_string = [] #string representation to export
matches_frames = []

dicts_attr_val = []
dicts_attr_val_string = []
dfs_matches_val = []
    
for val_idx in range(df_val.shape[0]):
        
    print(val_idx)
    attr_vector = attributions_val[val_idx,:]
    dict_attr_idx,df_matches_idx = get_attribution_for_hidden_layer(val_idx,df_val,'Molecule',attr_vector,df_substructures,neuron_thr=0.01,ismol=True,
                                     use_atom_weight=weighting_scheme)
    dicts_attr_val.append(dict_attr_idx)
    dicts_attr_val_string.append(dict_to_string(dict_attr_idx,'|'))
    dfs_matches_val.append(df_matches_idx)
df_matches_val = pd.concat(dfs_matches_val)
    
matches_frames.append(df_matches_val)
#attributions_atoms_val.append(dicts_attr_val)
#attribution_atoms_string.append(dicts_attr_val_string)

df_attributions = pd.DataFrame(data={'atom_attributions':dicts_attr_val_string}, index=df_val.index)

df_attributions.to_csv('../results/atom_attributions_ighidden_val.csv')

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
    df_evaluation.to_csv('../results/attribution_aucs_ighidden.csv')
