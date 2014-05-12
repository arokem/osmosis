import numpy as np

import osmosis.model.sparse_deconvolution as sfm
import osmosis.model.dti as dti

import osmosis.viz.mpl as mpl
import osmosis.utils as utils
import osmosis.snr as snr
import matplotlib
import osmosis.utils as ozu

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
    data, mask = obtain_data(data, mask)
        
    # Separate b values
    bval_list, bval_ind, unique_b, _ = ozu.separate_bvals(bvals)
    idx_array = np.arange(len(unique_b))
    
    new_bval_list = []
    for bi in idx_array:
        new_bval_list.append(bvals[bval_ind[bi]])
    
    # Add b = 0 values and indices to the other b values for tensor calculation
    bval_ind_wb0, bvals_wb0 = include_b0vals(idx_array, bval_ind, new_bval_list)
        
    # Find the property values for each grouped b values
    idx_mask = np.where(mask)
    log_prop = log_prop_vals(prop, saved_file, data, bvecs, idx_mask, idx_array, bval_ind_wb0, bvals_wb0, mask)
        
    # Fit a first order least squares solution to the specified property versus
    # b value to obtain slopes
    ls_fit = ls_fit_b(log_prop, unique_b/1000)
    
    # Display slopes of property on a mosaic
    slopeProp_all = disp_slopes(mask, ls_fit, prop)

    # Save all the values
    np.save('slope{0}_all.npy'.format(prop), slopeProp_all)
    np.save('log_{0}.npy'.format(prop), log_prop)
    np.save('ls_fit_{0}.npy'.format(prop), ls_fit[0])
    np.save('unique_b.npy', unique_b)
    
    return slopeProp_all, log_prop, ls_fit, unique_b

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
        #affine = data.get_affine()
        
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

def log_prop_vals(prop, saved_file, data, bvecs, idx_mask, idx_array, bval_ind_wb0, bvals_wb0, mask):
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
    prop_dict = {'FA':'FA', 'MD':'MD', 'mean diffusivity':'MD',
                'fractional anisotropy':'FA', 'dispersion index':'DI',
                'DI':'DI'}
    
    log_prop = list()
    for k in idx_array[:len(idx_array)-1]:
        if saved_file is 'no':
            params_file = 'temp'
        elif saved_file is 'yes':
            if prop_dict[prop] is 'DI':
                params_file = 'SparseDeconvolutionModel{0}.nii.gz'.format(k+1)
            else:
                params_file = 'TensorModel{0}.nii.gz'.format(k+1)
        if prop_dict[prop] is "FA":
            tensor_prop = dti.TensorModel(data[:,:,:,bval_ind_wb0[k]], bvecs[:,bval_ind_wb0[k]], bvals_wb0[k], mask = mask, params_file = params_file)
            prop_val = tensor_prop.fractional_anisotropy
            log_prop.append(np.log(prop_val[idx_mask] + 0.01))
        elif prop_dict[prop] is "MD":
            tensor_prop = dti.TensorModel(data[:,:,:,bval_ind_wb0[k]], bvecs[:,bval_ind_wb0[k]], bvals_wb0[k], mask = mask, params_file = params_file)
            prop_val = tensor_prop.mean_diffusivity
            log_prop.append(np.log(prop_val[idx_mask] + 0.01))
        elif prop_dict[prop] is 'DI':
            sfm_di = sfm.SparseDeconvolutionModel(data[:,:,:,bval_ind_wb0[k]], bvecs[:,bval_ind_wb0[k]], bvals_wb0[k], mask = mask, params_file = params_file)
            prop_val = sfm_di.dispersion_index()
            log_prop.append(prop_val[idx_mask] + 0.01)
            
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
    ls_fit: 1 dimensional array
        An array with the results from the least squares fit
    """
    if 0 in unique_b:
        unique_b = unique_b[1:]
        
    log_prop_matrix = np.matrix(log_prop)
    b_matrix = np.matrix([unique_b, np.ones(len(unique_b))]).T
    b_inv = utils.ols_matrix(b_matrix)
    ls_fit = np.dot(b_inv, log_prop_matrix)
    
    return ls_fit

def disp_slopes(mask, ls_fit, prop):
    """
    Prepares and displays the slopes in a mosaic.
    
    Parameters
    ----------
    mask: 3 dimensional array
        Brain mask of the data
    ls_fit: 1 dimensional array
        An array with the results from the least squares fit
        
    Returns
    -------
    slopeProp_all: 3 dimensional array
        Slope of the desired property across b values at each voxel
    """  
    prop_dict = {'FA':'FA', 'MD':'MD', 'mean diffusivity':'MD',
                'fractional anisotropy':'FA', 'dispersion index':'DI',
                'DI':'DI','SNR':'SNR', 'signal-to-noise ratio':'SNR',
                'signal to noise ratio':'SNR'}
    
    slopeProp_all = np.zeros_like(mask)
    slopeProp_all[np.where(mask)] = np.squeeze(np.array(ls_fit[0,:][np.isfinite(ls_fit[0,:])]))
    
    if prop_dict[prop] is 'SNR':
        fig = mpl.mosaic(slopeProp_all, cmap=matplotlib.cm.bone)
        fig.set_size_inches([20,10])
    else:
        fig = mpl.mosaic(slopeProp_all, cmap=matplotlib.cm.PuOr_r, vmin = -0.75, vmax = 0.75)
        fig.set_size_inches([20,10])
    
    return slopeProp_all
    
def sqrd_err(ls_fit, log_prop, unique_b, mask):
    """
    Calculates the squared error from the model fit
    
    Parameters
    ----------
    ls_fit: 1 dimensional array
        An array with the results from the least squares fit
    log_prop: list
        List of all the log of the desired property values
    unique_b: 1 dimensional array
        Array of all the unique b values found
        
    Returns
    -------
    sum_sqrd_err: 1 dimensional array
        An array with the results from the squared error calculations
    """
    if 0 in unique_b:
        unique_b = unique_b[1:]
            
    sqrd_err_list = list()
    ls_fit = np.array(ls_fit)
    for bi in np.arange(unique_b.shape[0]):
        fit_vals = ls_fit[0,:]*unique_b[bi]*np.ones([1,ls_fit.shape[1]]) + ls_fit[1,:]
        sqrd_err_list.append((log_prop[bi] - fit_vals)**2)
        
    sum_sqrd_err = np.squeeze(np.sum(np.array(sqrd_err_list).T, -1))
    
    disp_sqrd_err(sum_sqrd_err, mask)
    
    return sum_sqrd_err
    
def disp_sqrd_err(sum_sqrd_err, mask):
    """
    Prepares and displays the squared error in a mosaic.
    
    Parameters
    ----------
    sum_sqrd_err: 1 dimensional array
        An array with the sum squared error at each voxel
    mask: 3 dimensional array
        Brain mask of the data
        
    Returns
    -------
    sqrd_err_all: 3 dimensional array
        Squared error from model fit at each voxel
    """   
    sqrd_err_all = np.zeros_like(mask)
    sqrd_err_all[np.where(mask)] = sum_sqrd_err
    
    fig = mpl.mosaic(sqrd_err_all, cmap=matplotlib.cm.bone, vmax = 1)
    fig.set_size_inches([20,10])
    
    return sqrd_err_all
   
def scat_prop_snrSlope(log_prop, data, bvals, mask):
    """
    Displays a scatter density plot of the slopes of the log of the desired property
    values versus the slopes of the first order fit through SNR.
    
    Parameters
    ----------
    log_prop: list
        List of all the log of the desired property values
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    mask: 3 dimensional array
        Brain mask of the data
    """    
    bval_list, bval_ind, unique_b = snr.separate_bvals(bvals)
          
    ls_fit_bsnr = snr_ls_fit(data, bvals, mask, unique_b)
    ls_fit_prop = ls_fit_b(log_prop, unique_b)
    
    mpl.scatter_density(ls_fit_bsnr[0,:], ls_fit_prop[0,:])
    
def snr_ls_fit(data, bvals, mask, unique_b):
    """
    Fits a first order least squares solution to the SNR data versus b value
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    mask: 3 dimensional array
        Brain mask of the data
    unique_b: 1 dimensional array
        Array of all the unique b values found
        
    Returns
    -------
    ls_fit_bsnr: 1 dimensional array
        An array with the results from the least squares fit to the SNR data
    """
    if 0 in unique_b:
        unique_b = sort(unique_b)[1:]
        
    data, mask = obtain_data(data, mask)
    
    all_bsnr = list()         
    for bi in np.arange(len(unique_b)):
        all_bsnr.append(snr.b_snr(data, bvals, unique_b[bi], mask)[np.where(mask)])
    
    ls_fit_bsnr = ls_fit_b(all_bsnr, unique_b)
    
    return ls_fit_bsnr
    
def scat_prop_snr(log_prop, data, bvals, mask):
    """
    Displays a scatter density plot of SNR versus the slope of the desired property
    
    Parameters
    ----------
    log_prop: list
        List of all the log of the desired property values
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    mask: 3 dimensional array
        Brain mask of the data
    """    
    bval_list, bval_ind, unique_b, _ = ozu.separate_bvals(bvals)
    
    if 0 in unique_b:
        unique_b = unique_b[1:]
    
    bsnr = snr.b_snr(data, bvals, unique_b[0], mask)[np.where(mask)]
    ls_fit_prop = ls_fit_b(log_prop, unique_b)
    
    mpl.scatter_density(bsnr, ls_fit_prop[0,:])