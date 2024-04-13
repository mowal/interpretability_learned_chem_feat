from rdkit.Chem import Draw
import numpy as np
import io
from PIL import Image

def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img

def draw_mol_with_highlight_varying_colour(mol,attribution_dict,absolute_scale=True,absolute_limit=1,label_values=False):
    """
    function that saves a mol with atoms, but not bonds highlighted
    use attributions from dict to get highlight colours, all atoms highlighted
    mol: rdkit mol object
    highlightAtoms: list of atoms indexes (int)
    path: string
    filename: string
    returns None
    """
    if absolute_scale:
        highlight_colors = get_highlight_colors_absolute(attribution_dict,absolute_limit)
    else:
        highlight_colors = get_highlight_colors(attribution_dict)

    # Set the atom label to be the attribution value
    if label_values:
        for atom in mol.GetAtoms():
            if atom.GetIdx() in attribution_dict:
                value = round(attribution_dict.get(atom.GetIdx()), 3)
                atom.SetProp("atomNote", str(value))
    
    d2d = Draw.MolDraw2DCairo(350, 350)
    d2d.drawOptions().useBWAtomPalette()
    d2d.DrawMolecule(mol,highlightAtoms=list(attribution_dict.keys()),
                                             highlightAtomColors=highlight_colors,highlightBonds=[])
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()
 
    return png_data

def get_highlight_colors(attribution_dict):
    """
    function that converts attributions in corrresponding dict of RGB tuples
    highest positive (or negative) contribution 'normal' red (or blue)
    ratio of absolute value to max absolute values determines intensity of red/blue
    attribution_dict: dictionary atom_idx (int) --> attribution (float)
    """
    max_value = max(attribution_dict.values())
    min_value = min(attribution_dict.values())
    
    color_dict = {}
       
    abs_max = max(np.abs(max_value),np.abs(min_value))
        
    for atom in attribution_dict:
        intensity_prop = attribution_dict[atom]/abs_max
            
        if intensity_prop > 0:
            #R=1 get values for G,B
            g_b_value = 1 - intensity_prop
            color_dict[atom] = (1,g_b_value,g_b_value)
                
        else:
            #B=1 get vlaues for R,G
            r_g_value = 1 - np.abs(intensity_prop)
            color_dict[atom] = (r_g_value,r_g_value,1)
                
    return(color_dict)
    
def get_highlight_colors_absolute(attribution_dict,max_value,max_thresh=0.7):
    """
    function that converts attributions in corrresponding dict of RGB tuples
    highest positive (or negative) contribution 'normal' red (or blue)
    max_value: highest pos or neg atom attribution in dataset
    max_thresh: full colour intensity if that proportion of max_value is reached, gives clearer distinction of colours
    ratio of absolute value to max absolute values determines intensity of red/blue
    attribution_dict: dictionary atom_idx (int) --> attribution (float)
    """
    max_value_adjusted = max_value*max_thresh
    color_dict = {}
    if max_value>0:
        
        for atom in attribution_dict:

            if attribution_dict[atom] > max_value_adjusted:
                intensity_prop = 1

            else:
                intensity_prop = attribution_dict[atom]/max_value_adjusted

            if intensity_prop > 0:
                #R=1 get values for G,B
                g_b_value = 1 - intensity_prop
                color_dict[atom] = (1,g_b_value,g_b_value)

            else:
                #B=1 get vlaues for R,G
                r_g_value = 1 - np.abs(intensity_prop)
                color_dict[atom] = (r_g_value,r_g_value,1)
                
    else:
        for atom in attribution_dict:

            if attribution_dict[atom] < max_value_adjusted:
                intensity_prop = 1

            else:
                intensity_prop = attribution_dict[atom]/max_value_adjusted

            if intensity_prop > 0:
                #B=1 get values for R,G
                r_g_value = 1 - intensity_prop
                color_dict[atom] = (r_g_value,r_g_value,1)

            else:
                #R=1 get values for G,B
                g_b_value = 1 - np.abs(intensity_prop)
                color_dict[atom] = (1,g_b_value,g_b_value)
                
    return(color_dict)
