from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def map_attr_to_atoms(idx,df,col,attr_vector,radius,bits,ismol=False):
    """
    idx (int): index of compound in respective dataset
    df (pd.DataFrame): dataframe (train, val, test)
    attr_vector (np.ndarray): vector containing attributions for all bits, shape[2048,]
    ismol input: indicate if col is given as mol_col
    """
    if ismol:
        mol = df.loc[idx,col]
    else:
        mol = Chem.MolFromSmiles(df.loc[idx,col])
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, bitInfo=info)
    
    #create a dictionary (keys: atoms, values: summed attributions)
    dict_attr = {}
    for atom in mol.GetAtoms():
        dict_attr[atom.GetIdx()] = 0
    
    #get attributions only for bits set in the molecule, iterate through keys of info
    for bit in info.keys():
        
        bit_attr = attr_vector[bit]
        
        #get number of different atom envs corresponding to bit
        n_envs = len(info[bit])
        
        #normalise attribution to be assigned per bit by number of envs
        bit_attr_per_env = bit_attr/n_envs 
        
        #for each env separately distribute bit_attr_per_env to atoms and add to dict_attr
        for env_tup in info[bit]:
            if env_tup[1] == 0:
                amap = {env_tup[0]:0}
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol,env_tup[1],env_tup[0])
                amap={}
                submol=Chem.PathToSubmol(mol,env,atomMap=amap)
            n_atoms = len(amap)
            bit_attr_per_env_per_atom = bit_attr_per_env/n_atoms
            for atom in amap:
                dict_attr[atom]+=bit_attr_per_env_per_atom
                
    return(dict_attr)

def get_matches_tree(mol,root_smi,df_substr_neuron):
    """
    recursive function that searches tree for most specific substructure match(es)
    if a compound matches but none of its childs: add to set of substructures
    """
    match_set = set()
    remove_set = set()
    
    row = df_substr_neuron[df_substr_neuron['SMILES']==root_smi]
    
    #does it match root smi?
    root_mol = Chem.MolFromSmiles(root_smi,sanitize=False)
    if mol.HasSubstructMatch(root_mol):
        match_set.add(root_smi)
        
        if row.iloc[0,5] != set([0]): #direct_parents
            for parent in row.iloc[0,5]:
                parent_row = df_substr_neuron[df_substr_neuron['idx_in_neuron']==parent]
                
                if parent_row.shape[0]>0:
                    remove_set.add(parent_row.iloc[0,2]) #SMILES
        
        if len(row.iloc[0,6]) == 0: #direct_childs
            return(match_set,remove_set)
    
        else:
            for child in row.iloc[0,6]: #direct_childs
                child_row = df_substr_neuron[df_substr_neuron['idx_in_neuron']==child]
                match_set_rec,remove_set_rec = get_matches_tree(mol,child_row.iloc[0,2],df_substr_neuron) #SMILES col of child_row
                match_set = match_set.union(match_set_rec)
                remove_set = remove_set.union(remove_set_rec)
                
    return(match_set,remove_set)

def get_matches_neuron(mol,neuron,df_substr):
    """
    function that finds for a given mol and neuron matching substructures (most specific from each tree)
    mol = Chem.MolFromSmiles(smi)
    """
    df_substr_neuron = df_substr[df_substr['neuron']==neuron].copy()
    total_matches = set()
    
    #iterate through trees
    for tree in df_substr_neuron['tree'].unique():
        df_tree = df_substr_neuron[df_substr_neuron['tree']==tree]
        root_smi = df_tree[df_tree['direct_parents']==set([0])].iloc[0,2] #SMILES
        match,remove = get_matches_tree(mol,root_smi,df_tree)
        keep = match.difference(remove)
        total_matches = total_matches.union(keep)
        
    return(total_matches)

def get_attribution_for_hidden_layer(idx,df,col,attr_vector,df_substr,neuron_thr=0.01,ismol=False,
                                     use_atom_weight=False,adjust_atoms_nonmatches=False):
    """
    function to get atom attributions from hidden layer neuron attribution
    idx: index in validation set
    df: validation dataset containing mols or smiles
    col: columns containing molecules as mol or smiles
    attr_vector: attributions for compound idx
    df_substr: df containing substructures
    neuron_thr: threshold for inclusion of neurons (default: 0.01 of highest neuron attribution)
    is_mol: whether molecule is represented as mol in df
    use_atom_weight: whether weighting scheme for atoms is used
    adjust_atoms_nonmatches: whether attributions are assigned equally to all atoms if there is not match between test compound and extracted substructures for neuron
    """
    #lists for df_matches
    comp_list = []
    neuron_list = []
    neuron_attr_list = []
    idx_in_neuron_list = []
    smiles_list = []
    weight_list = []
    
    if ismol:
        mol = df.loc[idx,col]
    else:
        mol = Chem.MolFromSmiles(df.loc[idx,col])
        
    #create a dictionary (keys: atoms, values: summed attributions)
    dict_attr = {}
    for atom in mol.GetAtoms():
        dict_attr[atom.GetIdx()] = 0
        
    #get highest pos or neg attribution for neuron
    max_neuron_thr = max([np.abs(attr_vector.max()),np.abs(attr_vector.min())])
        
    #check if attribution for neuron is above thr
    neuron_thr_appl = max_neuron_thr * neuron_thr
    
    for neuron in df_substr['neuron'].unique():
        neuron_idx = int(neuron.split('_')[1])
        neuron_contrib = attr_vector[neuron_idx]
        
        if np.abs(neuron_contrib) < neuron_thr_appl:
            continue
        
        #get substructures for neuron
        substr_neuron = list(get_matches_neuron(mol,neuron,df_substr))
        
        if len(substr_neuron) == 0:
            if adjust_atoms_nonmatches:
                n_atoms = mol.GetNumAtoms()
                atom_bias = neuron_contrib/n_atoms
                
                for atom_idx in range(n_atoms):
                    dict_attr[atom_idx]+=atom_bias
                    
                continue
            else:
                continue
        
        #get atom attributions from neuron
        substr_contrib_general = neuron_contrib/len(substr_neuron)
        
        #iterate through substructures
        for substr in substr_neuron:
            
            df_substr_neuron = df_substr[(df_substr['neuron']==neuron)&(df_substr['SMILES']==substr)]
            #get df slice with one row per substr considered
            comp_list.append(idx)
            neuron_list.append(neuron)
            neuron_attr_list.append(neuron_contrib)
            idx_in_neuron_list.append(df_substr_neuron['idx_in_neuron'].iloc[0])
            smiles_list.append(substr)
            weight_list.append(df_substr_neuron['weight'].iloc[0])
            
            m_substr = Chem.MolFromSmiles(substr,sanitize=False)
            
            matches = mol.GetSubstructMatches(m_substr)
            
            #divide substr_contrib_general by occurrences of substr in mol
            substr_contrib_spec = substr_contrib_general/len(matches)
            
            for match in matches:
                if use_atom_weight:
                    for substr_atom_idx,atom in enumerate(match):
                        atom_weight = df_substr_neuron['atom_weights'].iloc[0][substr_atom_idx]
                        dict_attr[atom]+=substr_contrib_spec*atom_weight
                    
                else:
                    for atom in match:
                        dict_attr[atom]+=substr_contrib_spec/len(match) #normalise by number of atoms, treating all atoms equally
      
    df_matches = pd.DataFrame(data={'compound':comp_list,'neuron':neuron_list, 
                                    'neuron_attr':neuron_attr_list,
                                    'idx_in_neuron':idx_in_neuron_list,'SMILES':smiles_list,
                                   'weight':weight_list})
    
    return dict_attr,df_matches

def attribution_auc(true_atoms,attribution_dict):
    """
    function that computes attribution auc
    first get vectors for true and attributions sorted by atom index
    true atoms: list of integers
    attribution_dict: dictionary atom_idx (int) --> attribution (float)
    """
    attrib_vect = [attribution_dict[i] for i in range(len(attribution_dict))]
    true_vect = [1 if i in true_atoms else 0 for i in range(len(attribution_dict))]
    
    #special case: all atoms of mol true, then AUC not defined, return 'n/a'
    all_same = False
    
    if len(true_vect) == sum(true_vect):
        all_same = True
        
    elif sum(true_vect) ==0:
        all_same=True
    
    if not all_same:
        score = roc_auc_score(true_vect,attrib_vect)
    else:
        score = 'n/a'
    return score
