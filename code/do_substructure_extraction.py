import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from pytorch_classes_and_functions import DNN
from helper_functions import get_fp_matrix, create_df_activations, get_bit_comp_df
from fca_functions import fca_lattice_from_df, extract_substr_from_lattice
from substruct_tree_functions import get_tree_df,split_tree
import os

directory_specification = '../data/' #directory where specification file is stored
specification_name = 'ames_final'

df_specification = pd.read_csv(directory_specification+specification_name+'.csv', index_col=0)

#input variables from df_specifications
thr_comp = float(df_specification.loc['threshold_compounds','values']) #multiples of SD from mean to consider compounds for inclusion
thr_bits = float(df_specification.loc['threshold_bits','values']) #proportion of bits with highest weight to include
thr_fca_supp = float(df_specification.loc['threshold_fca_support','values']) #threshold to include concepts based on support
thr_fca_substr = float(df_specification.loc['threshold_fca_substructures','values']) #threshold to include substructures based on full support, as well as treshold to include extracted substructures per FC
thr_fca_weight_factor = float(df_specification.loc['threshold_fca_weight_factor','values']) #threshold to include concepts based on summed weight of bits, proportion of mean + thr_comp*SD
thr_fca_weight_factor2 = float(df_specification.loc['threshold_fca_weight_factor2','values']) #threshold to include concepts based on summed weight of bits for cases where thr_fca_supp is not reached, factor to multiply with weight_thr_fca                            
thr_fca_novelty = int(df_specification.loc['threshold_fca_novelty_limit','values']) #threshold to consider FCs based on occurrences of bits in previous FCs
fp_check = bool(df_specification.loc['fp_check','values']) #whether to use check if extracted substr is 'false positive' (acutally does not strongly activate neuron)

train_file = df_specification.loc['train_file','values']
train_file_type = df_specification.loc['train_file_type','values']
mol_column = df_specification.loc['mol_column','values']
activity_column = df_specification.loc['activity_column','values']

if train_file_type == 'smiles-csv':
    df_train = pd.read_csv(train_file, index_col=0)
    mols = [Chem.MolFromSmiles(smi) for smi in df_train[mol_column]]
    
elif train_file_type == 'mol-sdf':
    df_train = PandasTools.LoadSDF(train_file,molColName=mol_column,removeHs=False,strictParsing=False)
    mols = df_train[mol_column].to_list()
    
# change datatype in df_train
df_train = df_train.astype({activity_column:'int32'})

fps = [AllChem.GetMorganFingerprintAsBitVect(mol,radius=1, nBits=2048) for mol in mols]
fp_matrix_train = get_fp_matrix(fps)

#load model
model_file = df_specification.loc['model_file','values']
model_input = int(df_specification.loc['model_input','values'])
model_hidd_neurons = int(df_specification.loc['model_hidd_neurons','values'])
model_layers = int(df_specification.loc['model_layers','values'])
model_dropout = float(df_specification.loc['model_dropout','values'])


device = torch.device('cpu')
model = DNN(model_input,model_hidd_neurons, model_layers, model_dropout)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

#create list of arrays per layer, here weights just for first layer extracted
for name, param in model.named_parameters():
    if name == 'hiddens.0.weight':
        hidden_weight_arr = param.detach().numpy()
    elif name == 'output.weight':
        output_weight_arr = param.detach().numpy()

#get df with activations of all training compounds for all neurons in first hidden layer 
df_activations = create_df_activations(df_train,train_file_type,mol_column,activity_column,model,model_hidd_neurons)

#initialise a df to store all extracted substructures in
df_substr_final = pd.DataFrame(columns=['neuron','tree','SMILES','supporting_compounds','weight','direct_parents','direct_childs'])

#initialise a df to store all compounds and bits per neuron
df_comps_bits_final = pd.DataFrame(columns=['neuron','compounds','bits'])

neurons_list = list(df_activations.columns[:-1]) #list of all hidden neurons

#iterate through neurons
for neuron in neurons_list:
    print(neuron)
    neuron_l = int(neuron.split('_')[0]) #neuron layer
    neuron_p = int(neuron.split('_')[1]) #neuron position in layer
    
    #get df with all weights connecting input and neuron
    df_weights_neuron = pd.DataFrame(data=hidden_weight_arr[neuron_p,:],columns=['weights'])
    
    print('get compounds and bits')
    
    #get df of compounds and bits according to selected thresholds for inclusion
    df_comps_bits = get_bit_comp_df(neuron,df_activations,fp_matrix_train,df_weights_neuron,thr_bits,thr_comp)
    
    compound_list = [int(i) for i in df_comps_bits.index]
    
    #get map from compound ID to mol object
    mol_dict = {}
    
    for comp_idx in compound_list:
        if train_file_type == 'smiles-csv':
            mol_dict[comp_idx] = Chem.MolFromSmiles(df_train.loc[comp_idx,mol_column])
        elif train_file_type == 'mol-sdf':
            mol_dict[comp_idx] = df_train.loc[comp_idx,mol_column]
    
    print('FCA analysis')
    #get fca lattice 
    l = fca_lattice_from_df(df_comps_bits)
    
    if l==None: #no lattice obtained, empty object set, skip neuron
        print('no lattice obtained, empty object set for neuron, continue with next neuron')
        continue
    
    mean = np.mean(df_activations[neuron])
    std = np.std(df_activations[neuron])

    weight_thr_fca = thr_fca_weight_factor*(mean+thr_comp*std) #threshold for inclusion of FC based on summed weight of bits
    n_comps = df_comps_bits.shape[0]
    
    print('substructure extraction')
    
    #get df of extracted substr for all FCs
    df_substr = extract_substr_from_lattice(df_train,mol_column,thr_fca_supp,thr_fca_substr,weight_thr_fca,thr_fca_weight_factor2,df_weights_neuron,l,n_comps,train_file_type,thr_fca_novelty,model,neuron,mol_column,fp_check=fp_check)
    
    #store all unique substructures of neuron in single df
    tree_df = get_tree_df(df_substr)
    
    print('adding supp. compounds')
    
    #add supporting compounds from set of compounds
    for substr_idx,row in tree_df.iterrows():
        patt = Chem.MolFromSmiles(row['SMILES'],sanitize=False)
        
        for comp_idx in compound_list:
            if mol_dict[comp_idx].HasSubstructMatch(patt):
                row['supporting_compounds'].add(comp_idx)
                
    print('split tree')          
    #split tree by getting all roots with no more generic substructure
    tree_list = split_tree(tree_df)
    
    #store substructures of all tree_df from tree_list in single df, individual entries may be part of several trees (more preicse: subnetworks)
    for tree_idx,tree in enumerate(tree_list):
        tree_df_to_add = tree.copy()
        tree_df_to_add['neuron'] = [neuron for i in range(tree_df_to_add.shape[0])]
        tree_df_to_add['tree'] = [tree_idx for i in range(tree_df_to_add.shape[0])]
        
        df_substr_final = pd.concat([df_substr_final,tree_df_to_add])
        
    #get string of compounds and bits to store in df
    compound_string = ''
    for comp_idx in compound_list:
        compound_string+= '|{}'.format(comp_idx)
    compound_string = compound_string[1:] #remove initial '|'
    
    bit_string = ''
    for bit in df_comps_bits.columns:
        bit_string+= '|{}'.format(bit)
    bit_string = bit_string[1:] #remove initial '|'
    
    #store all considered compounds and bits for each neuron in a df
    df_comp_bit_to_add = pd.DataFrame(data={'neuron':[neuron],'compounds':[compound_string],'bits':[bit_string]})
    
    df_comps_bits_final = pd.concat([df_comps_bits_final,df_comp_bit_to_add])
    
run_description = df_specification.loc['run_description','values']

sucess=False
while not sucess:
    output_directory = '../results_'+run_description
    try:
        os.mkdir(output_directory)
        sucess=True
    except FileExistsError:
        print('Path already exists, enter new name for directory:')
        run_description = input()

#export dfs
df_substr_final.to_csv(output_directory+'/substructures.csv',index=False)
df_comps_bits_final.to_csv(output_directory+'/compounds_bits.csv',index=False)