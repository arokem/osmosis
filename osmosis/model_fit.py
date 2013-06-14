import numpy as np
import osmosis.model.dti as dti
import osmosis.viz.mpl as mpl
import osmosis.utils as utils
from osmosis.snr import separate_bvals
import matplotlib

def slope(data, bvals, bvecs, prop, mask):
    """
    Calculates and displays the slopes of a least squares solution fitted to either the
    log of the fractional anisotropy data or mean diffusivity data of the tensor model
    across the brain at different b values.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    prop: str
        String indicating the property to analyzed
        'FA': Fractional anisotropy
        'MD': Mean diffusivity
    mask: 3 dimensional array
        Brain mask of the data
        
    Returns
    -------
    slopeProp_all: 3 dimensional array
        Slope of the desired property across b values at each voxel
    """
    prop_dict = {'FA':'FA', 'MD':'MD', 'mean diffusivity':'MD', 'fractional anisotropy':'FA'}

    if hasattr(data, 'get_data'):
        data = data.get_data()
        affine = data.get_affine()
        
    if mask is not 'None':
        if hasattr(mask, 'get_data'):
            mask = mask.get_data()
        
    # Separate b values
    bval_list, bval_ind, unique_b = separate_bvals(bvals)
    idx_array = np.arange(len(unique_b))
    
    bval_ind_wb0 = list()
    # Add b values to indices for tensor model fitting
    for p in idx_array[1:]:
        bval_ind_wb0.append(np.concatenate((bval_ind[0], bval_ind[p])))
        
    # Find the property values for each grouped b values
    idx_mask = np.where(mask)
    log_prop = list()
    for k in idx_array[:len(unique_b)-1]:
        tensor_prop = dti.TensorModel(data[:,:,:,bval_ind_wb0[k]], bvecs[:,bval_ind_wb0[k]], bvals[bval_ind_wb0[k]], mask = mask, params_file='TensorModel{0}.nii.gz'.format(k+1))
        if prop_dict[prop] is "FA":
            prop_val = tensor_prop.fractional_anisotropy
            log_prop.append(np.log(prop_val[idx_mask] + 0.01))
        elif prop_dict[prop] is "MD":
            prop_val = tensor_prop.mean_diffusivity
            log_prop.append(np.log(prop_val[idx_mask] + 0.01))
    
    # Convert list into a matrix and make a matrix with b values.
    log_prop_matrix = np.matrix(log_prop)
    b_matrix = np.matrix([unique_b[1:], np.ones(len(unique_b[1:]))]).T
    inv = utils.ols_matrix(b_matrix)
    
    ls_fit = np.dot(inv, log_prop_matrix)
    
    #Calculates slopes    
    slopeProp_all = np.zeros_like(mask)
    slopeProp_all[idx_mask] = np.squeeze(np.array(ls_fit[0,:][np.isfinite(ls_fit[0,:])]))
    
    fig = mpl.mosaic(slopeProp_all, cmap=matplotlib.cm.bone)
    fig.set_size_inches([20,10])
    
    return slopeProp_all