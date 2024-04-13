import pandas as pd
from rdkit import Chem

def check_if_identical(tree_df,query_smi,weight,supporting_compounds):
    """
    function that checks if an identical substr is already in tree_df, only the one with higher summed weight is kept
    
    input: original tree_df, query_smi (str), weight(float): summed weight of FP bits, supporting_compounds (list of ints)
    returns: identical (Boolean), tree_df updated
    """
    
    identical=False
    query_mol = Chem.MolFromSmiles(query_smi,sanitize=False)
    for i,row in tree_df.iterrows():
        node_mol = Chem.MolFromSmiles(row['SMILES'],sanitize=False)
        #identical if both sub and superstructure
        if query_mol.HasSubstructMatch(node_mol) and node_mol.HasSubstructMatch(query_mol):
            identical =True
            #add supporting compounds and weight if higher
            row['supporting_compounds'] = row['supporting_compounds'].union(supporting_compounds)
            
            if weight > row['weight']:
                tree_df.loc[i,'weight'] = weight
                
            break #stop search if one identical was found
                
    return(identical,tree_df)

def get_specific_parents(query_smi,tree_df,id_0):
    
    """
    function finds most specific parents for query_smi in existing tree_df
    returns: set of parent IDs
    """
    #root as most specific parent initially (id_0=0 in call from get_tree_df), different from 0 in recursions
    query_mol = Chem.MolFromSmiles(query_smi,sanitize=False)
    smiles0 = tree_df.loc[id_0,'SMILES']
    mol0 = Chem.MolFromSmiles(smiles0,sanitize=False)
    parents = set()
    
    if query_mol.HasSubstructMatch(mol0) or len(smiles0)==0: #check that mol0 is more general (substructure) or the root (SMILES:'')
        
        for child_id in tree_df.loc[id_0,'direct_childs']:
            
            parents = parents.union(get_specific_parents(query_smi,tree_df,child_id))

        if len(parents) == 0:
            parents = set([id_0])
                
    return(parents)

def get_generic_children(query_smi,tree_df,id_0):
    """
    get most generic childre of query_smi in tree_df
    returns children (set of compound IDs)
    """
    query_mol = Chem.MolFromSmiles(query_smi,sanitize=False)
    #start looking at 'root structure'
    smiles0 = tree_df.loc[id_0,'SMILES']
    mol0 = Chem.MolFromSmiles(smiles0,sanitize=False)
    
    children = set()
    
    if mol0.HasSubstructMatch(query_mol):
        children.add(id_0)
        
    else:
        for child_id in tree_df.loc[id_0,'direct_childs']:
            
            children = children.union(get_generic_children(query_smi,tree_df,child_id))
            
            children_to_remove = set()
            for rem_check_child1_id in children:
                rem_check_smi1 = tree_df.loc[rem_check_child1_id,'SMILES']
                rem_check_mol1 = Chem.MolFromSmiles(rem_check_smi1,sanitize=False)
                
                for rem_check_child2_id in children:
                    if rem_check_child1_id == rem_check_child2_id:
                        continue
                    rem_check_smi2 = tree_df.loc[rem_check_child2_id,'SMILES']
                    rem_check_mol2 = Chem.MolFromSmiles(rem_check_smi2,sanitize=False)
                    if rem_check_mol2.HasSubstructMatch(rem_check_mol1):
                        children_to_remove.add(rem_check_child2_id)
            
            #get difference set between children and children_to_remove
            children = children.difference(children_to_remove)
            
    return(children) 

def add_supporting_compounds(idx,supporting_comps,tree_df):
    """
    for newly inserted substr, add ID as supporting to all parents
    """
    
    if len(tree_df.loc[idx,'direct_parents']) >0:
        for parent in tree_df.loc[idx,'direct_parents']:
            #add brackets to have a list with single set inside
            tree_df.loc[idx,'supporting_compounds'] = [tree_df.loc[idx,'supporting_compounds'].union(supporting_comps)]
            add_supporting_compounds(parent,supporting_comps,tree_df)
            
    return(tree_df)

def insert_row(smiles,supporting_comps,weight,atom_weights,spec_parents,gen_children,tree_df):
    """
    function unserts row of substructure into tree, takes car of hierarchy
    returns tree_df with row added
    """
    
    new_idx = max(list(tree_df.index))+1
    df_to_add = pd.DataFrame(data={'SMILES':smiles,'supporting_compounds':[supporting_comps],'weight':weight,'atom_weights':atom_weights,'direct_parents':[spec_parents],
                                                   'direct_childs':[gen_children]},index=[new_idx])
    tree_df =pd.concat([tree_df,df_to_add])
    
    #add as direct child to parents, remove direct childs from parents
    for parent in spec_parents:
        tree_df.loc[parent,'direct_childs'].add(new_idx)
        
        tree_df.loc[parent,'direct_childs'] = [tree_df.loc[parent,'direct_childs'].difference(gen_children)]
        
        #ensure set data type
        if type(tree_df.loc[parent,'direct_childs']) == int:
            tree_df.loc[parent,'direct_childs'] = [set([tree_df.loc[parent,'direct_childs']])]
        
    #add as parent to childs, remove parents from childs
    for child in gen_children:
        tree_df.loc[child,'direct_parents'].add(new_idx)
        
        tree_df.loc[child,'direct_parents'] = [tree_df.loc[child,'direct_parents'].difference(spec_parents)]
        
    #add supporting compounds to all parents, recursive function
    tree_df = add_supporting_compounds(new_idx,supporting_comps,tree_df)
    
    return(tree_df)

def get_tree_df(df_substruct):
    """
    function that stores all unique extrcated susbtructures for neuron in network ('tree') structure
    
    input: df_substruct: df coming from extract_substr_from_lattice function in fca_fucntions.py
    returns: pd.DataFrame with information about substructures and hierarchical relations
    """
    tree_df = pd.DataFrame(columns=['SMILES','supporting_compounds','weight','atom_weights','direct_parents','direct_childs'])
    
    #add empty SMILES as root
    tree_df = pd.concat([tree_df,pd.DataFrame(data={'SMILES':'','supporting_compounds':set(),'weight':0,'atom_weights':{},'direct_parents':set(),'direct_childs':set()},
                                                    index=[0])])
    
    for i,row in df_substruct.iterrows():
        query_smi = row['SMILES']
        supp_comps = row['supporting_compounds']
        weight = row['weight']
        atom_weights = row['atom_weights']
        #check if identical substructure in tree
        identical,tree_df = check_if_identical(tree_df,query_smi,weight,supp_comps)
        if identical == False:
            
            #get most specific parents
            parents = get_specific_parents(query_smi,tree_df,0)
            
            #get most generic child(s)
            children = get_generic_children(query_smi,tree_df,0)
            
            #insert row
            tree_df = insert_row(query_smi,supp_comps,weight,atom_weights,parents,children,tree_df)
            
    tree_df['idx_in_neuron'] = list(tree_df.index)
        
    return(tree_df)
       
def find_tree_childs(tree_df,idx):
    """
    function that finds childs of all levels given an idx for the respective root
    returns set of child IDs
    """
    childs = set()
    
    for child_idx in tree_df.loc[idx,'direct_childs']:
        childs.add(child_idx)
        if len(tree_df.loc[child_idx,'direct_childs'])>0:
            childs = childs.union(find_tree_childs(tree_df,child_idx))
            
    return(childs)

#function to separate trees on level1
def split_tree(tree_df):
    """
    function splits complete network into 'trees' (or subnetworks) where root is existing substructure
    for which no more generic substructure exists
    returns list of split tree_dfs
    """
    tree_list = []

    #create separate df for each child of the root
    tree_roots = tree_df.loc[0,'direct_childs']
    
    for comp_idx in tree_roots:
        childs = find_tree_childs(tree_df,comp_idx) #find childs IDs in tree_df to root, one ID may belong to different trees 
        childs.add(comp_idx)
        df = tree_df.loc[childs,:].copy()
        tree_list.append(df)
    
    return(tree_list)
