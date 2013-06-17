import numpy as np
import osmosis.model.dti as dti
import osmosis.viz.mpl as mpl
import osmosis.utils as utils
from osmosis.snr import separate_bvals
import matplotlib

def slope(data, bvals, bvecs, prop, mask = 'None', saved_file = 'yes'):
    """
    Calculates and displays the slopes of a least squares solution fitted to either
    the log of the fractional anisotropy data or mean diffusivity data of the tensor
    model across the brain at different b values.
    
    Parameters
    ----------
    data: 4 dimensional array or Nifti1Image
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    prop: str
        String indicating the property to analyzed
        'FA': Fractional anisotropy
        'MD': Mean diffusivity
    mask: 3 dimensional array or Nifti1Image
        Brain mask of the data
    saved_file: 'str'
        Indicate whether or not you want the function to create or use saved
        parameter files
        'no': Function will not create or use saved files
        'yes': Function will create or use saved files
        
    Returns
    -------
    slopeProp_all: 3 dimensional array
        Slope of the desired property across b values at each voxel
    """
    
    # Making sure inputs are all in the right format for calculations
    prop_dict = {'FA':'FA', 'MD':'MD', 'mean diffusivity':'MD', 'fractional anisotropy':'FA'}
    data, mask = obtain_data(data, mask)
        
    # Separate b values
    bval_list, bval_ind, unique_b = separate_bvals(bvals)
    idx_array = np.arange(len(unique_b))
    
    # Add b = 0 values and indices to the other b values for tensor calculation
    bval_ind_wb0, bvals_wb0 = include_b0vals(idx_array, bval_ind, bval_list)
        
    # Find the property values for each grouped b values
    idx_mask = np.where(mask)
    log_prop = log_prop_vals(prop, saved_file, data, bvecs, idx_mask, idx_array)
    
    # Convert list into a matrix and make a matrix with b values.
    ls_fit = ls_fit_b(log_prop, unique_b)
    
    #Plot slopes    
    slopeProp_all = plot_slopes(mask, ls_fit)
    
    return slopeProp_all

def obtain_data(data, mask):
    """
    Gets the data from the diffusion data and mask inputs if they're not already in
    array form.  If inputs are already in array form, the function simply returns
    the original inputs.
    
    Parameters
    ----------
    data: 4 dimensional array or Nifti1Image
        Diffusion MRI data
    mask: 3 dimensional array or Nifti1Image
        Brain mask of the data
        
    Returns
    -------
    data: 4 dimensional array
        Diffusion MRI data
    mask: 3 dimensional array
        Brain mask of the data
    """
    if hasattr(data, 'get_data'):
        data = data.get_data()
        affine = data.get_affine()
        
    if mask is not 'None':
        if hasattr(mask, 'get_data'):
            mask = mask.get_data()
            
    return data, mask
def include_b0vals(idx_array, bval_ind, bval_list):
    """
    Tensor model calculations work better if given b = 0 values and indices.  This
    function concatenates the values and indices to the b = 0 values to the other
    b value and indices arrays.
    
    Parameters
    ----------
    idx_array: 1 dimensional array
        Array with sequential numbers from 0 to the length of the number of b values
        for indexing over.
    bval_ind: list
        List of the indices corresponding to the separated b values.  Each index
        contains an array of the indices to the grouped b values with similar values
    bval_list: list
        List of separated b values.  Each index contains an array of grouped b values
        with similar values
        
    Returns
    -------
    bval_ind_wb0: list
        List of the indices corresponding to the non-zero separated b values
        concatenated to the indices of the b = 0 values.
    bvals_wb0: list
        List of the values corresponding to the non-zero separated b values
        concatenated to the valuese of the b = 0 values.
    """
    bval_ind_wb0 = list()
    bvals_wb0 = list()
    for p in idx_array[1:]:
        bval_ind_wb0.append(np.concatenate((bval_ind[0], bval_ind[p])))
        bvals_wb0.append(np.concatenate((bval_list[0], bval_list[p])))
        
    return bval_ind_wb0, bvals_wb0

def log_prop_vals(prop, saved_file, data, bvecs, idx_mask, idx_array):
    """
    Tensor model calculations of the given property
    
    Parameters
    ----------
    prop: str
        String indicating the property to analyzed
        'FA': Fractional anisotropy
        'MD': Mean diffusivity
    saved_file: 'str'
        Indicate whether or not you want the function to create or use saved
        parameter files
        'no': Function will not create or use saved files
        'yes': Function will create or use saved files
    data: 4 dimensional array
        Diffusion MRI data
    bvecs: 3 dimensional array
        All the b vectors
    idx_array: ndarray
        Array with the indices indicating the location of the non-zero values within
        the mask
        
    Returns
    -------
    log_prop: list
        List of all the log of the desired property values
    """

    log_prop = list()
    for k in idx_array[:len(idx_array)-1]:
        if saved_file is 'no':
            params_file = 'temp'
        elif saved_file is 'yes':
            params_file = 'TensorModel{0}.nii.gz'.format(k+1)
        tensor_prop = dti.TensorModel(data[:,:,:,bval_ind_wb0[k]], bvecs[:,bval_ind_wb0[k]], bvals_wb0[k], mask = mask, params_file = params_file)
        if prop_dict[prop] is "FA":
            prop_val = tensor_prop.fractional_anisotropy
            log_prop.append(np.log(prop_val[idx_mask] + 0.01))
        elif prop_dict[prop] is "MD":
            prop_val = tensor_prop.mean_diffusivity
            log_prop.append(np.log(prop_val[idx_mask] + 0.01))
    
    return log_prop
    
def ls_fit_b(log_prop, unique_b):
    """
    Does calculations for fitting a first order least squares solution to the
    properties
    
    Parameters
    ----------
    log_prop: list
        List of all the log of the desired property values
    unique_b: 1 dimensional array
        Array of all the unique b values found
        
    Returns
    -------
    log_prop: list
        List of the indices corresponding to the non-zero separated b values
        concatenated to the indices of the b = 0 values.
    """
    log_prop_matrix = np.matrix(log_prop)
    b_matrix = np.matrix([unique_b[1:], np.ones(len(unique_b[1:]))]).T
    b_inv = utils.ols_matrix(b_matrix)
    ls_fit = np.dot(b_inv, log_prop_matrix)
    
    return ls_fit

def plot_slopes(mask, ls_fit):
    """
    Does calculations for fitting a first order least squares solution to the
    properties
    
    Parameters
    ----------
    log_prop: list
        List of all the log of the desired property values
    unique_b: 1 dimensional array
        Array of all the unique b values found
        
    Returns
    -------
    slopeProp_all: 3 dimensional array
        Slope of the desired property across b values at each voxel
    """
    idx_mask = np.where(mask)
    slopeProp_all = np.zeros_like(mask)
    slopeProp_all[idx_mask] = np.squeeze(np.array(ls_fit[0,:][np.isfinite(ls_fit[0,:])]))
    
    fig = mpl.mosaic(slopeProp_all, cmap=matplotlib.cm.bone)
    fig.set_size_inches([20,10])
    
    return slopeProp_all