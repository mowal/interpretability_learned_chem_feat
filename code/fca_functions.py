from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from concepts import Context
from fp_check_functions import get_mean_activation_compound_set
from helper_functions import dict_to_string

def fca_lattice_from_df(df):
    """
    function that creates fca lattice from df containing comps and bits 
    input: pd.DataFrame with compounds as objects (rows) and FP bits as properties (columns)
    returns lattice object from concepts package (if successful)
    """
    
    #get df in format required by concepts package
    df.replace(0,np.nan,inplace=True)
    df.replace(1,'X',inplace=True)
    
    objects = df.index.tolist() #define objects
    objects_str = [str(obj) for obj in objects]
    properties = list(df) #define properties
    bools = list(df.fillna(False).astype(bool).itertuples(index=False, name=None))
    
    try:
        c = Context(objects_str, properties, bools)
        lattice = c.lattice
        return(lattice)
    except ValueError:
        return(None)


def get_highlight_atoms_from_fc(concept,df_train,smiles_col,train_file_type):
    """
    for compounds in extent of FC: determine which atoms are part of FP bits in intent
    NOTE: function needs to be adapted to work for different FP types
    
    concept: FC from lattice object
    df_train: train dataset (pd.DataFrame)
    smiles_col: column with smiles or mol data (string)
    train_file_type: smiles-csv or mol-sdf (string)
    
    returns: pd.DataFrame containing compounds of FC as rows;
    columns: SMILES (string), highlight_atoms (list of atoms that are part of FP bits), atom_to_bit (dictionary mapping atom ID to bit IDs) 
    
    """
    
    #get bits as integer
    bits = [int(i.split('_')[1]) for i in concept.intent]
    
    #initialise lists for return df
    highlight_atoms = []
    smiles = []
    list_d_atom_bits = []
    
    #iterate through compounds to get highlight atoms
    for comp_id in concept.extent:
        comp_id = int(comp_id)
        atoms_mol = set()
        d_atom_bits = {} #store mapping from atom to bits in dict
        
        if train_file_type == 'smiles-csv':
            smi = df_train.loc[comp_id,smiles_col]
            smiles.append(smi)
            mol = Chem.MolFromSmiles(smi)
        elif train_file_type == 'mol-sdf':
            mol = df_train.loc[comp_id,smiles_col]
            smi = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smi) # atom numbering same as when creating mol from smiles in rdkit
            smiles.append(smi)   
            
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=1, nBits=2048,bitInfo=bit_info)
        
        for bit in bits:
            for tup in bit_info[bit]:
                    atoms_mol.add(tup[0])
                    if tup[1]==1:
                        atom = mol.GetAtomWithIdx(tup[0])
                        for atom2 in atom.GetNeighbors():
                            atoms_mol.add(atom2.GetIdx())
                            
        highlight_atoms.append(atoms_mol)
        
        #get mapping atom -> bits 
        for atom in atoms_mol:
            d_atom_bits[atom] = set()
        
        for bit in bits:
            for tup in bit_info[bit]:
                atom_idx = tup[0]
                if atom_idx in atoms_mol:
                    d_atom_bits[atom_idx].add(bit)
                if tup[1] == 1: #add neighbour atoms distance 1 (i.e. radius of FP)
                    atom_obj = mol.GetAtomWithIdx(atom_idx)
                    for ne_atom in atom_obj.GetNeighbors():
                        ne_atom_idx = ne_atom.GetIdx()
                        if ne_atom_idx in atoms_mol:
                            d_atom_bits[ne_atom_idx].add(bit)
                    
        list_d_atom_bits.append(d_atom_bits)
        
    #return as df
    df_hightlight_atoms = pd.DataFrame(index=[int(i) for i in concept.extent])
    df_hightlight_atoms['SMILES'] = smiles
    df_hightlight_atoms['highlight_atoms'] = highlight_atoms
    df_hightlight_atoms['atom_to_bit'] = list_d_atom_bits
    return(df_hightlight_atoms)

def get_highlight_atom_fragments(df_highlight_atoms):
    """
    function that gets atoms of connected fragmets
    
    input: df_highlight_atoms (=output from get_highlight_atoms_from_fc)
    
    returns: df_highlight_atoms with one col added ('highlight_atom_fragments')
            'highlight_atom_fragments' is a list where each element is a set of atom ids that form a connected substructure within the compound  
    """
    
    fc_substruct_atoms = []
    
    for compound,row in df_highlight_atoms.iterrows():
        compound_substruct_atoms = []
        mol = Chem.MolFromSmiles(row['SMILES'])      
        for idx in row['highlight_atoms']:
            #get all neighbour atoms
            atom = mol.GetAtomWithIdx(idx)
            #figure out if atom is already part of a substructure
            substruct_idx = None 
            
            for i,substructs in enumerate(compound_substruct_atoms):
                if idx in substructs:
                    substruct_idx = i
            
            if substruct_idx != None:
                continue #jump to next iteration if atom (and therefore all its neighbours) were already added
    
            if substruct_idx == None:
                compound_substruct_atoms.append(set([idx]))
                substruct_idx = len(compound_substruct_atoms)-1

            #add neighbour atoms (and neighbours of neighbours etc) of atom if they are in 
            neighbours_level1 = []
            atoms_to_be_added = set()
            for x in atom.GetNeighbors():
                ne_idx = x.GetIdx()
                if ne_idx not in row['highlight_atoms']: #neighbour atom already in highlight_atoms
                    continue
                elif ne_idx in atoms_to_be_added: #neighbour atom already flagged to be added
                    continue

                neighbours_level1.append(ne_idx)

            if len(neighbours_level1) ==0:
                continue
     
            atoms_to_be_added = atoms_to_be_added.union(set(neighbours_level1))

            #get neigbours level 2 and more (while loop until no more additional atoms are added)
            neighbours_level2 = []
            counter = 0 #counter to distinguish loop 0 from following
            len_end = 0
            
            while True:
                if counter ==0:
                    len_start = 0
                else:
                    len_start = len_end

                for ne_idx in neighbours_level1:
                    atom2 = mol.GetAtomWithIdx(ne_idx)
                    for y in atom2.GetNeighbors():
                        ne2_idx = y.GetIdx()

                        if ne2_idx not in row['highlight_atoms']:
                            continue
                        elif ne2_idx in atoms_to_be_added: #atom already flagged to be added
                            continue

                        neighbours_level2.append(ne2_idx)
                
                if len(neighbours_level2) ==0:
                    break

                #break if no atoms added after one cycle
                len_end = len(neighbours_level2)
                if len_end == len_start:
                    break

                atoms_to_be_added = atoms_to_be_added.union(set(neighbours_level2))
                neighbours_level1 = neighbours_level2.copy()
                counter+=1 
        
            #add all found atoms to fragment
            compound_substruct_atoms[-1] = compound_substruct_atoms[-1].union(atoms_to_be_added)
            
        fc_substruct_atoms.append(compound_substruct_atoms)
        
    df_highlight_atoms['highlight_atom_fragments'] = fc_substruct_atoms
    return(df_highlight_atoms)

def get_top_fragments(df_fragments,df_weights):
    """
    function that for each mol extracts the fragment with the highest weight
    
    input: df_fragments (output from get_highlight_atom_fragments)
    df_weights: data table containing weights of bits for neuron (pd.DataFrame)
    
    returns: df_top_fragments, same number of rows as df_fragments
            columns: 'SMILES' (string, SMILES of whole compound), 'fr_SMILES' (string, SMILES of Top fragment of compound),
                    'fr_atoms' (set, atom ids included in Top fragment), 'fr_weight' (float, summed weight of included bits),
                    'atom_weights' (dictionary, relative weight of atoms based on weight bits, sum to 1)
    
    """
    #initialise lists for export df
    top_fr_smiles = []
    top_fr_atoms_list = []
    top_fr_weight = []
    top_fr_atom_weights_list = []
    
    for comp_id, row in df_fragments.iterrows():
        fr_weights = []
        #find fragment with highest weight
        for frag_idx,frag_atoms in enumerate(row['highlight_atom_fragments']):
            frag_bits = set()
            for frag_atom in frag_atoms:
                frag_bits = frag_bits.union(row['atom_to_bit'][frag_atom])
                
            #sum weights of bits to get weight of fragment
            fr_weights.append(0)
            for bit in frag_bits:
                fr_weights[-1]+=df_weights.loc[bit,'weights']
                
        top_fr_weight.append(max(fr_weights))
        top_fr_idx = fr_weights.index(max(fr_weights)) #index within list of fragments per compound
        top_fr_atoms = row['highlight_atom_fragments'][top_fr_idx]
        top_fr_atoms_list.append(top_fr_atoms)
        
        #get smiles of top fragment
        mol = Chem.MolFromSmiles(row['SMILES'])
        break_bonds = set([]) #bonds that need to be broken
        
        for atom_id in top_fr_atoms_list[-1]:
            atom = mol.GetAtomWithIdx(atom_id)
            
            #iterate through neighbour atoms
            for ne_atom in atom.GetNeighbors():
                if ne_atom.GetIdx() not in top_fr_atoms_list[-1]: #break bond if neighbour atom not part of fragment
                    break_bonds.add(mol.GetBondBetweenAtoms(atom_id,ne_atom.GetIdx()).GetIdx())
                    
        if len(break_bonds)>0: #only not true if every atom in mol is highlight atom
            mol_fr = Chem.rdmolops.FragmentOnBonds(mol,bondIndices=break_bonds,addDummies=False)

            #first: return with atom idx to find fragment of interest
            fragments_as_atom_id = Chem.rdmolops.GetMolFrags(mol_fr)

            for i,fragment in enumerate(fragments_as_atom_id):
                if fragment[0] in top_fr_atoms_list[-1]:
                    fragment_id = i

            #return as mol objects and take fragment
            fragments_tuple = Chem.rdmolops.GetMolFrags(mol_fr,asMols=True,sanitizeFrags=False)
            target_fragment = fragments_tuple[fragment_id]
            
        else:
            target_fragment = mol

        #cave: atom numbering in fragment may be different compared to parent compound
        top_fr_smiles.append(Chem.MolToSmiles(target_fragment))
        
        #target_fragment is relevant fragment as mol object, get atom mapping between mol and target_fragment
        top_fr_atom_abs_weights = {} #store atom contributions with idx from fragment
        
        matches = mol.GetSubstructMatches(target_fragment)
        correct_match = False
        
        for match in matches:
            if set(match) != row['highlight_atom_fragments'][top_fr_idx]:
                continue
            else:
                correct_match = True
                if correct_match:
                    break #match now is the correct match for code below
        
        #get weight of fragment (i.e. summed weight of included bits)
        for frag_atom_id,mol_atom_id in enumerate(match): #atom idx in fragment vs full mol
            top_fr_atom_abs_weights[frag_atom_id] = 0
            
            #sum for each atom weights of bits
            for bit in row['atom_to_bit'][mol_atom_id]:
                top_fr_atom_abs_weights[frag_atom_id]+=df_weights.loc[bit,'weights']
                
        #transform top_fr_atom_abs_weights to relative weights, must add up to 1 per fragment
        sum_weights_fragment = sum(top_fr_atom_abs_weights.values())
        top_fr_atom_rel_weights = {}
        
        for frag_atom_id in top_fr_atom_abs_weights:
            top_fr_atom_rel_weights[frag_atom_id] = round(top_fr_atom_abs_weights[frag_atom_id]/sum_weights_fragment, 3)
        
        
        string_top_fr_atom_rel_weights = dict_to_string(top_fr_atom_rel_weights, '|')
        top_fr_atom_weights_list.append(string_top_fr_atom_rel_weights)
                 
    #create df
    df_top_fragments = pd.DataFrame(index=df_fragments.index)
    df_top_fragments['SMILES'] = df_fragments['SMILES']
    df_top_fragments['fr_SMILES'] = top_fr_smiles
    df_top_fragments['fr_atoms'] = top_fr_atoms_list
    df_top_fragments['fr_weight'] = top_fr_weight
    df_top_fragments['atom_weights'] = top_fr_atom_weights_list
    
    return(df_top_fragments)
    

def get_top_fragment_summary(df_top_fragments):
    """
    function that aggregates entrys for repeated fragments and stores addtional info (substructure-superstructure, support incl. substr)
    
    input: df_top_fragments (output from get_top_fragments)
    
    returns: df_top_fragment_summary
            columns: 'fr_SMILES' (string: SMILES of unique fragment), 'fr_comps' (set: compound ids from which fragment was extracted),
                    'fr_weight' (float: summed weight of fragment from FP bits), 'fr_identical_support' (float: compounds from which fragment was extracted divided by all compounds in extent),
                    'fr_parents' (set: parent fragments, i.e. substructures), 'fr_childs' (set: child fragments, i.e. superstructures),
                    'fr_full_support' (float: 'fr_identical_support' + support of child fragments)
    """
        
    unique_frags = []
    unique_frag_comps = []
    unique_frag_weight = []
    
    #find unique fragments, get comp_idx and weight
    for comp_idx,row in df_top_fragments.iterrows():
        if len(unique_frags)==0: #initialise first fragment
            unique_frags.append(row['fr_SMILES'])
            unique_frag_comps.append(set([int(comp_idx)]))
            unique_frag_weight.append(row['fr_weight'])
            continue
           
        #check if identical with any existing fragment
        identical=False
        query_mol = Chem.MolFromSmiles(row['fr_SMILES'],sanitize=False)
        for i,u_frag in enumerate(unique_frags):
            u_mol = Chem.MolFromSmiles(u_frag,sanitize=False)
                
            #identical if query_mol is both sub- and superstructure
            if u_mol.HasSubstructMatch(query_mol) and query_mol.HasSubstructMatch(u_mol):
                unique_frag_comps[i].add(int(comp_idx))
                identical=True
                break
        
        if identical==False:
            #add as new fragment    
            unique_frags.append(row['fr_SMILES'])
            unique_frag_comps.append(set([comp_idx]))
            unique_frag_weight.append(row['fr_weight'])
                    
    #get support for identical match for each fragment
    ident_support = []
    for comp_set in unique_frag_comps:
        ident_support.append(len(comp_set)/df_top_fragments.shape[0])
    
    #check sub/superstr relation between fragments, use nested for loops
    unique_frag_parents = [set() for i in range(len(unique_frags))] #one set element per fragment
    unique_frag_childs = [set() for i in range(len(unique_frags))]  #one set element per fragment
    
    for i,fr1 in enumerate(unique_frags):
        m1 = Chem.MolFromSmiles(fr1,sanitize=False)
        for j,fr2 in enumerate(unique_frags):
            if i>=j: #makes sure that each pair is evaluated only once and identical frags never
                continue
            
            m2 = Chem.MolFromSmiles(fr2,sanitize=False)
            if m1.HasSubstructMatch(m2):
                unique_frag_parents[i].add(j)
                unique_frag_childs[j].add(i)
            
            elif m2.HasSubstructMatch(m1):
                unique_frag_parents[j].add(i)
                unique_frag_childs[i].add(j)
                          
    #get support including childs
    full_support = []
    
    for ident_supp,childs in zip(ident_support,unique_frag_childs):
        full_support.append(ident_supp+len(childs)/df_top_fragments.shape[0])
        
        
    #get summary df
    df_top_fragment_summary = pd.DataFrame(data={'fr_SMILES':unique_frags,'fr_comps':unique_frag_comps,
                                                 'fr_weight':unique_frag_weight,'fr_identical_support':ident_support,
                                                'fr_parents':unique_frag_parents,'fr_childs':unique_frag_childs,
                                                'fr_full_support':full_support})
    #add atom weights
    atom_weight_list = []
    
    for fra_smi in df_top_fragment_summary['fr_SMILES']:
        #get atom contrib from frag with highest weight (to avoid cases where fragment did not account for all bits)
        atom_weight_list.append(df_top_fragments[df_top_fragments['fr_SMILES']==fra_smi].sort_values(by='fr_weight',ascending=False)['atom_weights'].iloc[0])
    
    df_top_fragment_summary['atom_weights'] = atom_weight_list
    
    return(df_top_fragment_summary)

def extract_substr_from_lattice(df_train,smiles_col,support_thr_fc,support_thr_substr,weight_thr,weight_thr2_factor,df_weights_neuron,lattice,n_comps,train_file_type,threshold_bit_occurrence,model,neuron,mol_col,fp_check=False,limit_substr=200):
    """
    function going from FCA lattice to extracted substructures
    
    df_train: train dataset (pd.DataFrame)
    smiles_col: column with smiles or mol data (string)
    support_thr_fc: threshold for inclusion of FCs based on support (float)
    support_thr_substr: threshold for inclusion of substr based on support (float)
    weight_thr: threshold for inclusion of FC based on summed weight of bits (float)
    weight_thr2_factor: for FCs not meeting support_thr_fc, use this factor to multiply with weight_thr to allow inclusion (float)
    df_weights_neuron: data table containing weights of bits for neuron (pd.DataFrame)
    lattice: FCA lattice for neuron (lattice object from concepts package)
    n_comps: number of compounds for neuron (int)
    train_file_type: smiles-csv or mol-sdf (string)
    limit_substr: stop extraction once certain number of substructures has been extracted (int, default:200)
    
    return: pd.DataFrame with columns: SMILES of substr (string), supporting compounds including childs (set object), weight (float: summed weight of bits included in fragment),
            atom_weights: dictionary (atom-ID -> atom-weight), FC_ID: identifier for each FC (int)
    """
    
    bit_counts = {} #dict to store how often each bit has been included in a FC
    
    
    weight_thr2 = weight_thr*weight_thr2_factor
    
    #intialise lists to store content to be returned in df
    substr_smiles = []
    supporting_comps = []
    weights = []
    fc_ids = []
    atom_weights = []
 
    for i,concept in enumerate(reversed(lattice)): #reversed to start with FCs with highest compound support
        extent,intent=concept
        
        #check if concept above threshold for weight and support
        concept_support = len(extent)/n_comps
        
        concept_weight = 0
        for bit in intent:
            bit_n = int(bit.split('_')[1])
            concept_weight+= df_weights_neuron.loc[bit_n,'weights']
            
        if concept_support < support_thr_fc and concept_weight < weight_thr2:
            continue
        
        if concept_weight < weight_thr:
            continue
            
        #check if all bits were already used at least threshold_bit_occurrence times
        if bit_thresh_violated(bit_counts,intent,threshold_bit_occurrence): 
            continue
        
        #add bits to bit_counts to keep track from bit_thresh_violated function
        for bit in intent:
            if bit not in bit_counts:
                bit_counts[bit] = 1
            else:
                bit_counts[bit]+=1
        
        df_ha = get_highlight_atoms_from_fc(concept,df_train,smiles_col,train_file_type)
        df_haf = get_highlight_atom_fragments(df_ha)
        df_top_fragments = get_top_fragments(df_haf,df_weights_neuron)
        df_top_fragments_summ = get_top_fragment_summary(df_top_fragments)

        #get all fragments above weigth threshold and support threshold
        for j,row in df_top_fragments_summ.iterrows():
            if row['fr_weight']>weight_thr and row['fr_full_support']>support_thr_substr:
                
                supporting_set = row['fr_comps']
                for child_row in row['fr_childs']:
                    supporting_set = supporting_set.union(df_top_fragments_summ.loc[child_row,'fr_comps'])
                
                #get mols from supporting_set
                mols = [df_train.loc[idx,mol_col] for idx in supporting_set]
                
                if fp_check:
                    
                    mean_act_fragment = get_mean_activation_compound_set(row['fr_SMILES'],mols,neuron,model,fp_radius=1,fp_bits=2048) #NOTE: function needs to be adapted to work for different FP type
                    if  mean_act_fragment < weight_thr:
                        continue #if fragment failed to pass: move on to next
                      
                substr_smiles.append(row['fr_SMILES'])
                
                supporting_comps.append(supporting_set)
                weights.append(row['fr_weight'])
                atom_weights.append(row['atom_weights'])
                fc_ids.append(i)
                
        #check number of unique substructures, if > limit_substr: break
        n_substr = len(set(substr_smiles))
        if n_substr>=limit_substr:
            break

    #store in df
    df_substructures = pd.DataFrame(data={'SMILES':substr_smiles,'supporting_compounds':supporting_comps,
                                         'weight':weights,'atom_weights':atom_weights,'FC_ID':fc_ids})
     
    return(df_substructures)

def bit_thresh_violated(bit_counts,intent,thr):
    """
    function to check if all bits in intent have been included in intent at least thr times. If so --> violated
    
    input:
        bit_counts (dictionary, bit_id: count how many FCs with this bit have been considered before for neuron)
        intent (intent object from contents package for current FC)
        thr: (integer: hyperparameter of extraction workflow)
    
    returns boolean: True if violated (then FC is ignored), False if not violated (then FC is considered)
    """
    violated = True
    for bit in intent:
        if bit not in bit_counts:
            violated=False
            break
        elif bit_counts[bit] <thr:
            violated = False
            break
    
    return violated
    