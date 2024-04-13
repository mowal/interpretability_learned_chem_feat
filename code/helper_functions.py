import pandas as pd
import numpy as np
from pytorch_classes_and_functions import AmesDataset

def create_df_activations(df_train,train_file_type,mol_column,activity_column,model,model_hidd_neurons):
    """
    function to create df_activations: n rows (training compounds), m columns (hidden neurons)
    assumes 1 hidden layer, adapt if more than 1 hidden layer
    
    input: df_train (pd.DataFrame), train_file_type (string), mol_column(string),activity_column (string),
    model (instance of DNN), model_hidd_neurons: numbe rof neurons in first hidden layer (int)
    output: pd.DataFrame or None (if not train_file_type == 'smiles-csv')
    """
    if train_file_type == 'smiles-csv':
        train_dataset = AmesDataset(df_train,smiles_col=mol_column,fingerprint_size=2048,fp_radius=1)
    elif train_file_type == 'mol-sdf':
        print('implement different dataset object to create train_dataset')
        return None
        
    activations_list = model.hidden_activations(train_dataset.x) #get all activations
    activations_first = activations_list[0] #get activations of first hidden layer
    
    columns = ['1_{}'.format(i) for i in range(model_hidd_neurons)]
    df_activations = pd.DataFrame(data=activations_first,columns=columns)
    return df_activations

def get_fp_matrix(fp_list): 
    """
    convert list of fps to a numpy matrix
    input: list of FPs
    returns: np.ndarray, rows: compounds, columns: FP bits
    """
    fp_matrix = np.empty(shape=(len(fp_list),len(fp_list[0])))
    for row,fp in enumerate(fp_list):
        for col,bit in enumerate(fp):
            fp_matrix[row,col] = bit
    return fp_matrix

def get_bit_comp_df(neuron,df_activations,fp_train,df_weights_neuron,bit_thr,comp_thr_sd): 
    """
    function that creates df of bits and compounds given the respective thresholds
    input:
    neuron: e.g. 1_45 (string)
    df_activations (pd.DataFrame)
    fp_train: np.ndarray
    df_weights_neuron: df with all weights connecting input and neuron (pd.DataFrame)
    bit_thr: threshold for inclusion of bits (float)
    comp_thr_sd: threshold for inclusion of compounds (float)
    
    returns: df (pd.DataFrame) with n rows (included compounds) and m columns (included FP bits) 
    """
    #get compounds above mean
    mean  = np.mean(df_activations[neuron])
    std = np.std(df_activations[neuron])
    neuron_comp_idx = list(df_activations[df_activations[neuron]> mean+comp_thr_sd*std][neuron].index)
    
    n_bits = int(df_weights_neuron.shape[0] * bit_thr)
    neuron_top_bits = list(df_weights_neuron.sort_values(by='weights',ascending=False).index[:n_bits])
    
    fp_matrix_top_comp = fp_train[neuron_comp_idx,:].copy()
    fp_matrix_top = fp_matrix_top_comp[:,neuron_top_bits]
    df_comps_bits = pd.DataFrame(data=fp_matrix_top,columns=['bit_{}'.format(i) for i in neuron_top_bits],
                                 index=neuron_comp_idx)
    
    #filter out bits with zero support in top compounds
    to_drop = []
    for col in df_comps_bits.columns:
        if np.sum(df_comps_bits[col]) ==0:
            to_drop.append(col)
        
    df_comps_bits.drop(to_drop,axis=1,inplace=True)
    
    return(df_comps_bits)

def convert_atom_weights_col(df_substr):
    """
    function to convert atom_weights column back to dictionaries as required for df_subtructures
    """
    
    new_col_list = []
    for weight_string in df_substr['atom_weights']:
        new_col_list.append({})
        for element in weight_string.split('|'):
            for i,element2 in enumerate(element.split(':')):
                if i==0:
                    atom_idx = int(element2)
                elif i==1:
                    weight = float(element2)
            new_col_list[-1][atom_idx] = weight
    
    df_substr['atom_weights'] = new_col_list
    
    return df_substr

def dict_to_string(d,sep):
    """
    function to convert a dict into a string representation to store in a csv
    """
    if sep==':':
        raise Exception('separator must not be ":"')
    string = ''
    for key,value in zip(d.keys(),d.values()):
        string+='{}:{}{}'.format(key,value,sep)
        
    #remove terminal separator
    string = string[:-1]
    
    return string

def string_to_dict(s,sep,key_type,value_type):
    """
    function to convert as string to a dictionary
    s: string
    sep: separator
    key_type: dtype for key
    value_type: dtype for value
    returns dict
    """
    d = {} 
    for element1 in s.split(sep):
        for i,element2 in enumerate(element1.split(':')):
            if i==0:
                key = element2
            else:
                value = element2
        d[key_type(key)] = value_type(value)
        
    return d

def string_to_set(string):
    """
    function to convert string back to a set from a csv file
    """
    
    set_to_return = set([])
        
    if string != 'set()':
        string_stripped = string[1:-1] #remove curly brackets
        for element in string_stripped.split(','):
            set_to_return.add(int(element))
            
    return(set_to_return)
    
def string_to_int_list(string,sep):
    """
    function to get a list from a string with a separator
    """
    l = []
    for element in string.split(sep):
        l.append(int(element))
        
    return l