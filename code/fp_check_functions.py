from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import combinations
import numpy as np
import torch

def get_neighbor_atoms(m,substr_atoms,radius=1):
    """
    get set of atoms not in substr, but in range of radius (dependent on radius)
    
    m: mol object of considered compound
    substr_atoms: atom indexes for match of substr with m
    radius: radius of atom environments used in Morgan FP
        
    returns set of neighbor atoms (defined above), one for each distance from 1 to radius
    """
    substr_atoms_set = set(substr_atoms)
    neighbor_atoms_dict = {}
    
    for r in range(1,radius+1):
        neighbor_atoms_dict[r] = set([])
    
    for substr_atom in substr_atoms:
        amap={}
        env = Chem.FindAtomEnvironmentOfRadiusN(m,radius,substr_atom)
        submol=Chem.PathToSubmol(m,env,atomMap=amap)
        for atom in amap.keys():
            neighbor_atoms_dict[r].add(atom)
            
    neighbor_atoms_dict[r] = neighbor_atoms_dict[r].difference(substr_atoms_set)
    
    return neighbor_atoms_dict

def get_always_and_optional_bits(m,substr_atoms,neighbor_atoms_dict,radius,info):
    """
    get set of always on (all implied atoms of env within substr)
    and optional bits (not all implied atoms of env within substr)
    
    m: mol object of considered compound
    substr_atoms: atom indexes for match of substr with m
    neighbor_atoms_dict: key: r (1,...,radius) -> value (set of neighbour atoms for r)
    radius: radius of atom environments used in Morgan FP
    info: bit_info of FP for m
    
    """
    always_bits = set([])
    for bit in info:
        for env in info[bit]:
            if env[0] in substr_atoms:
                always_bits.add(bit)
                break
    
    optional_bits = set([])
    
    for r in range(1,radius+1):
        neighbor_atoms = neighbor_atoms_dict[r]
        for bit in info:
            for env in info[bit]:
                if env[0] in neighbor_atoms and env[1] == r:
                    if bit not in always_bits:
                        optional_bits.add(bit)
                        break
    
    return always_bits, optional_bits

def get_fp_vector(bits,nbits=2048):
    """
    create a vector of length nbits given a set of on-bits
    """
    fp_vector = np.zeros(nbits)
    for bit in bits:
        fp_vector[bit] = 1
    
    return fp_vector

def get_fp_permutations(always_bits,optional_bits):
    """
    from set of always_bits and optional_bits get permutations of Morgan FP to find activations
    """
    fps = []
    
    for l in range(len(optional_bits)+1):
        for comb in combinations(optional_bits,l):
            combined_bits = always_bits.union(set(comb))
            fps.append(get_fp_vector(combined_bits))
            
    return fps

def get_mean_activation_fps(model,neuron,fp_list):
    """
    for a collection of (fragment) FPs: get mean activation of neuron by model
    """
    neuron_layer = int(neuron.split('_')[0])-1
    neuron_id = int(neuron.split('_')[1])
    
    #convert fp_list to torch.tensor
    fps_np = np.array(fp_list)
    fps_torch = torch.tensor(fps_np,dtype=torch.float)
    
    #get mean for neuron of interest
    activations_all_layers = model.hidden_activations(fps_torch)
    activations_layer = activations_all_layers[neuron_layer]
    mean_activation_neuron = np.mean(activations_layer[:,neuron_id])   
    
    return mean_activation_neuron

def get_mean_activation_compound(substr,m,neuron,model,fp_radius=1,fp_bits=2048,compound_type='smi'):
    """
    for compound (m) with fragment (substr): get average activation for different FP bit combination vectors
    bits complete in substructure are always on, bits partially in susbtructure are permuted
    """
    if compound_type =='smi':
        m = Chem.MolFromSmiles(m)
        
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(m,fp_radius,nBits=fp_bits,bitInfo=info)
    
    #get matches of m with m_substr
    m_substr = Chem.MolFromSmiles(substr,sanitize=False)
    substr_atoms = m.GetSubstructMatch(m_substr)
    
    neighbor_atoms_dict = get_neighbor_atoms(m,substr_atoms,radius=fp_radius)
    always_bits,optional_bits = get_always_and_optional_bits(m,substr_atoms,neighbor_atoms_dict,fp_radius,info)
    fp_list = get_fp_permutations(always_bits,optional_bits)
    mean_activation = get_mean_activation_fps(model,neuron,fp_list)
    
    return mean_activation

def get_mean_activation_compound_set(substr,mols,neuron,model,fp_radius=1,fp_bits=2048,compound_type='mol'):
    """
    for set of compounds (mols): get mean activation for fragment (substr)
    
    input:
    substr (string): SMILES of substructure to check
    mols: list of rdkit.Mol or strings
    neuron: stirng, identifier of neuron
    model: pytorch model with hidden_activations method (see pytorch_classes_and_functions.py)
    fp_radius: int
    fp_bits: int
    compound_type: 'mol' or 'smi'
    
    """
    if type(mols[0]) ==str:
        mols = [Chem.MolFromSmiles(smi) for smi in mols]
    
    mean_single_comps = [] #activation for single comps
    for m in mols:
        mean_single = get_mean_activation_compound(substr,m,neuron,model,fp_radius=fp_radius,fp_bits=fp_bits,compound_type=compound_type)
        mean_single_comps.append(mean_single)
    
    mean_compound_set = np.mean(mean_single_comps)
    
    return mean_compound_set
