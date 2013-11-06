"""

This module contains various functions associated with leave n out cross-validation.
All functions use the multi_bvals' and/or SFM's predict function.

"""
import numpy as np
from math import factorial as f
import itertools
import time
import os
import inspect
import nibabel as nib
import osmosis.utils as ozu
from osmosis.utils import separate_bvals

import osmosis.model.sparse_deconvolution as sfm
import osmosis.model.dti as dti

import osmosis.snr as snr
import osmosis.multi_bvals as sfm_mb
import osmosis.snr as snr

def partial_round(bvals, factor = 1000.):
    """
    Round only the values about equal to zero.
    """
    partially_rounded = bvals
    for j in np.arange(len(bvals/factor)):
        if round(bvals[j]/factor) == 0:
            partially_rounded[j] = 0
            
    return partially_rounded
def create_combos(bvecs, bvals_pool, data, these_b_inds, these_b_inds_rm0, all_inc_0, vec_pool, num_choose, combo_num):
    """
    Helper function for the cross-validation functions
    """
    
    idx = list(these_b_inds_rm0)
    these_inc0 = list(all_inc_0)
    
    low = (combo_num)*num_choose
    high = np.min([(combo_num*num_choose + num_choose), len(vec_pool)])
    
    vec_pool_inds = vec_pool[low:high]
    vec_combo = these_b_inds[vec_pool_inds]
    vec_combo_rm0 = these_b_inds_rm0[vec_pool_inds]
        
    # Remove the chosen indices from the rest of the indices
    for choice_idx in vec_pool_inds:
        these_inc0.remove(these_b_inds[choice_idx])
        idx.remove(these_b_inds_rm0[choice_idx])
        
    # Make the list back into an array
    these_inc0 = np.array(these_inc0)
    
    # Isolate the b vectors, b values, and data not including those
    # to be predicted
    these_bvecs = bvecs[:, these_inc0]
    these_bvals = bvals_pool[these_inc0]
    this_data = data[:, :, :, these_inc0]
    
    # Need to sort the indices first before indexing full model's
    # regressors
    si = sorted(idx)
    
    return si, vec_combo, vec_combo_rm0, vec_pool_inds, these_bvecs, these_bvals, this_data, these_inc0
    
def new_mean_combos(vec_pool_inds, data, bvals, bvecs, mask, ad, rd, over_sample,
                    bounds, solver, mean, b_inds, b_idx1, b_idx2):
    """
    Helper function for calculating a new mean from all b values and corresponding data
    """
    
    inds_list = list(np.arange(len(bvals)))
    
    # Remove combo indices from the indices from all the b values.
    for vc in vec_pool_inds:
        inds_list.remove(b_inds[1:][b_idx1][vc])
        if b_idx2 is not None:
            inds_list.remove(b_inds[1:][b_idx2][vc])

    inds_arr = np.array(inds_list)
    
    # New data with removed indices.
    fit_all_data = data[..., inds_arr]
    fit_all_bvecs = bvecs[:, inds_arr]
    fit_all_bvals = bvals[inds_arr]
    
    # Now put this into SFM multi_b class in order to grab the calculated mean
    mod = sfm_mb.SparseDeconvolutionModelMultiB(fit_all_data, fit_all_bvecs, fit_all_bvals,
                                                mask = mask, axial_diffusivity = ad,
                                                radial_diffusivity = rd,
                                                over_sample = over_sample,
                                                bounds = bounds, solver = solver,
                                                mean = mean, params_file = "temp")
    _, b_inds_ar, _, _ = separate_bvals(fit_all_bvals, mode = "remove0")
	
    sig_out, new_params = mod.fit_flat_rel_sig_avg

    return sig_out, new_params, b_inds_ar[b_idx1]

def predict_n(data, bvals, bvecs, mask, ad, rd, n, b_mode, b_idx1 = 0, mean = None,
              b_idx2 = None, over_sample=None, bounds = None, new_mean = None, solver=None):
    """
    Predicts signals for a certain percentage of the vertices.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    n: int
        Integer indicating the percent of vertices that you want to predict
    b_mode: str
        'all': if fitting to all b values
        'bvals': if fitting to individual b values
        
    Returns
    -------
    actual: 2 dimensional array
        Actual signals for the predicted vertices
    predicted: 2 dimensional array 
        Predicted signals for the vertices left out of the fit
    """ 
    t1 = time.time()
    
    # Separating b values and grabbing indices
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
    
    _, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = separate_bvals(bvals,
                                                                    mode = 'remove0')
    # Grab the indices where b is not equal to 0
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    all_b_idx_rm0 = np.arange(len(all_b_idx))
    
    predicted = np.empty((int(np.sum(mask)), len(all_b_idx)))
    
    # Generate the regressors in the full model from which we choose the regressors in
    # the reduced model from.  This is so you won't have to recalculate the regressors
    # each time you do cross-validation.
    if b_mode is "all": 
        full_mod = sfm_mb.SparseDeconvolutionModelMultiB(data, bvecs, bvals,
                                                        mask = mask,
                                                        axial_diffusivity = ad,
                                                        radial_diffusivity = rd,
                                                        over_sample = over_sample,
                                                        bounds = bounds,
                                                        solver = solver,
                                                        params_file = "temp")
        indices = np.array([b_idx1])
    elif b_mode is "bvals":
        if b_idx2 is None:
            #mean = "empirical"
            indices = np.arange(len(unique_b[1:]))
        else:
            # Order of predicted: b_idx1 to b_idx1, b_idx1 to b_idx2, b_idx2 to b_idx2
            # b_idx2 to b_idx1
            indices = np.array([b_idx1, b_idx2])
            predicted11 = np.empty((int(np.sum(mask)),) + (len(b_inds[1]),))
            predicted12 = np.empty(predicted11.shape)
            predicted21 = np.empty(predicted11.shape)
            predicted22 = np.empty(predicted11.shape)

    for bi in indices:
        if b_mode is "all":
            all_inc_0 = np.arange(len(rounded_bvals))
            # Indices are locations of all non-b = 0
            these_b_inds = all_b_idx
            these_b_inds_rm0 = all_b_idx_rm0
        elif b_mode is "bvals":
            all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][bi]))
            these_b_inds = b_inds[1:][bi]
            these_b_inds_rm0 = b_inds_rm0[bi]
        bvals_pool = bvals
        vec_pool = np.arange(len(these_b_inds))
        
        # Need to choose random indices so shuffle them!
        np.random.shuffle(vec_pool)
        
        # How many of the indices are you going to leave out at a time?
        num_choose = (n/100.)*len(these_b_inds)
        
        if np.mod(100, n):
            e = "Data not equally divisible by %d"%n
            raise ValueError(e)
        elif abs(num_choose - round(num_choose)) > 0:
            e = "Number of directions not equally divisible by %d"%n
            raise ValueError(e)
        
        for combo_num in np.arange(np.floor(100./n)):
            (si, vec_combo, vec_combo_rm0,
            vec_pool_inds, these_bvecs, these_bvals,
            this_data, these_inc0) = create_combos(bvecs, bvals_pool, data,
                                                    these_b_inds,
                                                    these_b_inds_rm0,
                                                    all_inc_0, vec_pool,
                                                    num_choose, combo_num)                
                
            mod = sfm_mb.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                         mask = mask, axial_diffusivity = ad,
                                                         radial_diffusivity = rd,
                                                         over_sample = over_sample,
                                                         bounds = bounds, solver = solver,
                                                         mean = mean, params_file = "temp")
            if new_mean is not None:
                # Get the parameters from fitting a mean model to all b values not including
                # the ones left out
                if (b_mode is "bvals") & (b_idx2 == None):
                    b_mean1 = bi
                elif b_idx2 is not None:
                    b_mean1 = b_idx1
                sig_out, new_params, b_inds_ar = new_mean_combos(vec_pool_inds, data, bvals, bvecs,
                           mask, ad, rd, over_sample, bounds, solver, mean, b_inds, b_mean1, b_idx2)
            else:
                new_params = None

            if b_mode is "bvals":
                mod.fit_flat_rel_sig_avg = [sig_out[:, b_inds_ar], new_params]

            # Grab regressors from full model's preloaded regressors.  This only works if
            # not predicting across b values.
            if (b_idx2 == None) & (b_mode is "all"):
                fit_to = full_mod.regressors[0][:, si]
                if over_sample is None:
                    tensor_regressor = full_mod.regressors[1][:, si][si, :]
                else:
                    tensor_regressor = full_mod.regressors[1][si, :]
                fit_to_demeaned = full_mod.regressors[2][:, si]
                fit_to_means = full_mod.regressors[3]

            if b_idx2 != None:
                vec_combo_rm0 = vec_pool_inds
                if bi == b_idx1:
                    predicted12[:, vec_pool_inds] = mod.predict(bvecs[:, b_inds[1:][b_idx2][vec_pool_inds]],
                                                                 bvals[b_inds[1:][b_idx2][vec_pool_inds]], 
                                                                 new_params = new_params)[mod.mask]
                    predicted11[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                                                 new_params = new_params)[mod.mask]
                else:
                    predicted21[:, vec_pool_inds] = mod.predict(bvecs[:, b_inds[1:][b_idx1][vec_pool_inds]],
                                                                 bvals[b_inds[1:][b_idx1][vec_pool_inds]], 
                                                                 new_params = new_params)[mod.mask]
                    predicted22[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                                                 new_params = new_params)[mod.mask]
            else:
                predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                                            new_params = new_params)[mod.mask]          
    t2 = time.time()
    print "This program took %4.2f minutes to run"%((t2 - t1)/60)
    
    if b_idx2 != None:
        actual1 = data[mod.mask][:, b_inds[1:][b_idx1]]
        actual2 = data[mod.mask][:, b_inds[1:][b_idx2]]
        return actual1, actual2, predicted11, predicted12, predicted22, predicted21
    else:
        actual = data[mod.mask][:, all_b_idx]
        return actual, predicted
    
def predict_grid(data, bvals, bvecs, mask, ad, rd, n, over_sample = None, solver = None, bounds = None):
    """
    Predicts signals for a certain percentage of the vertices with all b values.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    n: int
        Integer indicating the percent of vertices that you want to predict
    b_mode: str
        'all': if fitting to all b values
        'bvals': if fitting to individual b values
        
    Returns
    -------
    actual: 2 dimensional array
        Actual signals for the predicted vertices
    predicted: 2 dimensional array 
        Predicted signals for the vertices left out of the fit
    """ 
    t1 = time.time()
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
    _, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = separate_bvals(bvals,
                                                                    mode = 'remove0')
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    all_b_idx_rm0 = np.arange(len(all_b_idx))
    
    actual = np.empty((np.sum(mask), len(all_b_idx)))
    predicted = np.empty(actual.shape)
    
    b1k_inds = np.arange(len(b_inds[1]))
    b2k_inds = np.arange(len(b_inds[2]))
    b3k_inds = np.arange(len(b_inds[3]))
    
    np.random.shuffle(b1k_inds)
    np.random.shuffle(b2k_inds)
    np.random.shuffle(b3k_inds)
   
    for bi in np.arange(3):

        group = [b1k_inds[bi*30:(bi*30+30)], b2k_inds[bi*30:(bi*30+30)], b3k_inds[bi*30:(bi*30+30)]] 

        these_b_inds = np.concatenate((b_inds[1][group[0]],
                                        b_inds[2][group[1]],
                                        b_inds[3][group[2]]))
        these_b_inds_rm0 = np.concatenate((b_inds_rm0[0][group[0]],
                                            b_inds_rm0[1][group[1]],
                                            b_inds_rm0[2][group[2]]))
        all_inc_0 = np.concatenate((b_inds[0], these_b_inds))
                        
        bvals_pool = bvals
        vec_pool = np.arange(len(these_b_inds))
        
        # Need to choose random indices so shuffle them!
        np.random.shuffle(vec_pool)
        
        # How many of the indices are you going to leave out at a time?
        num_choose = (n/100.)*len(these_b_inds)
        
        if np.mod(100, n):
            e = "Data not equally divisible by %d"%n
            raise ValueError(e)
        elif abs(num_choose - round(num_choose)) > 0:
            e = "Number of directions not equally divisible by %d"%n
            raise ValueError(e)
 
        for combo_num in np.arange(np.floor(100./n)):
            (si, vec_combo, vec_combo_rm0,
            these_bvecs, these_bvals,
            this_data, these_inc0) = create_combos(bvecs, bvals_pool, data,
                                                                  these_b_inds,
                                                                  these_b_inds_rm0,
                                                                  all_inc_0, vec_pool,
                                                                  num_choose, combo_num)
            
            mod = sfm_mb.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                        mask = mask, params_file = "temp",
                                                        axial_diffusivity = ad,
                                                        radial_diffusivity = rd,
                                                        bounds = bounds,
                                                        over_sample = over_sample,
                                                        solver = solver)
                                       
            predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo])[mod.mask]
            actual[:, vec_combo_rm0] = data[mod.mask][:, vec_combo]
            
    t2 = time.time()
    print "This program took %4.2f minutes to run"%((t2 - t1)/60)
    
    return actual, predicted
    
def demean(fit_to, tensor_regressor, mod):
    """
    This function demeans the signals and tensor regressors.
    
    Parameters
    ----------
    fit_to: 2 dimensional array
        Original signal fitted to.  Size should be equal to the number of voxels by the
        number of directions.
    tensor_regressor: 2 dimensional array
        The predicted signals from the tensor model.  Size should be equal to the number
        of directions fit to by the number of directions fit to.
        
    Returns
    -------
    fit_to: 2 dimensional array
        Original signal fitted to.  Size should be equal to the number of voxels by the
        number of directions.
    tensor_regressor: 2 dimensional array
        The predicted signals from the tensor model.  Size should be equal to the number
        of directions fit to by the number of directions fit to.
    design_matrix: 2 dimensional array
        Demeaned tensor regressors
    fit_to_demeaned: 2 dimensional array
        Demeaned signal fitted to
    fit_to_means:
        The means of the original signal fitted to.
    """
    
    fit_to_demeaned = np.empty(fit_to.shape)
    fit_to_means = np.empty(fit_to.shape)
    design_matrix = np.empty(tensor_regressor.shape)
           
    for bidx in np.arange(len(mod.unique_b)):
        for vox in xrange(mod._n_vox):
            # Need to demean everything across the vertices that were fitted to
            fit_to_demeaned[vox, mod.b_inds_rm0[bidx]] = (fit_to[vox, mod.b_inds_rm0[bidx]]
                                                - np.mean(fit_to[vox, mod.b_inds_rm0[bidx]]))
            fit_to_means[vox, mod.b_inds_rm0[bidx]] = np.mean(fit_to[vox, mod.b_inds_rm0[bidx]])

            tr = np.squeeze(tensor_regressor[mod.b_inds_rm0[bidx]])                
            design_matrix[mod.b_inds_rm0[bidx]] = (tr
                                    - np.mean(tensor_regressor[mod.b_inds_rm0[bidx]].T, -1))
                                    
    return [fit_to, tensor_regressor, fit_to_demeaned, fit_to_means]

    

def kfold_xval(model_class, data, bvecs, bvals, k, mask, b_mode = "all", **kwargs):
    """
    Predicts signals for a certain percentage of the vertices.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    k : int 
       The number of folds. Divide the data into k equal parts.
    b_mode: str
        'all': if fitting to all b values
        'bvals': if fitting to individual b values
    
    Returns
    -------
    actual: 2 dimensional array
        Actual signals for the predicted vertices
    predicted: 2 dimensional array 
        Predicted signals for the vertices left out of the fit
    """ 
    t1 = time.time()
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
    _, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = separate_bvals(bvals,
                                                                    mode = 'remove0')
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    all_b_idx_rm0 = np.arange(len(all_b_idx))
    partially_rounded = partial_round(bvals)
    
    actual = np.empty((np.sum(mask), len(all_b_idx)))
    predicted = np.empty(actual.shape)
    
    # Generate the regressors in the full model from which we choose the regressors in
    # the reduced model from.
    if b_mode is "all": 
        indices = np.array([0])
    elif b_mode is "bvals":
        indices = np.arange(len(unique_b[1:]))
    
    for bi in indices:
        if b_mode is "all":
            all_inc_0 = np.arange(len(rounded_bvals))
            these_b_inds = all_b_idx
            these_b_inds_rm0 = all_b_idx_rm0
        elif b_mode is "bvals":
            all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][bi]))
            these_b_inds = b_inds[1:][bi]
            these_b_inds_rm0 = b_inds_rm0[bi]
        bvals_pool = partially_rounded
        vec_pool = np.arange(len(these_b_inds))
        
        # Need to choose random indices so shuffle them!
        np.random.shuffle(vec_pool)
        
        # How many of the indices are you going to leave out at a time?
        num_choose = k

        if np.mod(len(these_b_inds), k): 
            raise ValueError("The number of directions are not equally divisible by %d"%k)
 
        for combo_num in np.arange(len(these_b_inds)/k):
            (si, vec_combo, vec_combo_rm0,
            these_bvecs, these_bvals,
            this_data, these_inc0) = create_combos(bvecs, bvals_pool, data,
                                                                  these_b_inds,
                                                                  these_b_inds_rm0,
                                                                  all_inc_0, vec_pool,
                                                                  num_choose, combo_num)
            
            mod = model_class(this_data, these_bvecs, these_bvals,
                                  mask = mask, params_file = "temp", **kwargs)
                                                 
            if len(inspect.getargspec(mod.predict)[0]) > 2:
                predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo])[mod.mask]
            elif len(inspect.getargspec(mod.predict)[0]) == 2:
                predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo])[mod.mask]
            
            actual[:, vec_combo_rm0] = data[mod.mask][:, vec_combo]

    t2 = time.time()
    print "This program took %4.2f minutes to run"%((t2 - t1)/60)
    return actual, predicted
    
def predict_bvals(data, bvals, bvecs, mask, ad, rd, b_idx1, b_idx2, n = 10,
                  solver = "nnls", mode = None):
    """
    Predict for each b value.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    b_idx1: int
        Unique b value index of the b value to fit to.
    b_idx2: int
        Unique b value index of the b value to predict.
        
    Returns
    -------
    actual: 2 dimensional array
        Actual signals for the predicted vertices
    predicted: 2 dimensional array 
        Predicted signals for the vertices left out of the fit
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
    bval_list_rm0, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = separate_bvals(bvals,
                                                                     mode = 'remove0')
    all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][b_idx1]))
   
    # Get data and indices of the directions to predict.
    predict_inds = b_inds[1:][b_idx2]
    actual = data[np.where(mask)][:, predict_inds]
    
    if mode == "kfold_xval":
        [actual1, actual2, predicted11,
        predicted12, predicted22, predicted21] = predict_n(data, bvals, bvecs, mask, ad, rd, n,
                                                           b_mode = "bvals", b_idx1 = b_idx1,
                                                           b_idx2 = b_idx2, new_mean = "Yes",
                                                           solver = solver)
    else:
        mod = sfm_mb.SparseDeconvolutionModelMultiB(data[:,:,:,all_inc_0],
                                                    bvecs[:,all_inc_0],
                                                    bvals[all_inc_0],
                                                    mask = mask,
                                                    axial_diffusivity = ad,
                                                    radial_diffusivity = rd,
                                                    solver = solver,
                                                    params_file = 'temp')
        predicted11 = mod.predict(bvecs[:, predict_inds], bvals[predict_inds])[mod.mask]
        actual1 = actual
        
    return actual1, actual2, predicted11, predicted12, predicted22, predicted21
    
def nchoosek(n,k):
    """
    Finds all the number of unique combinations from choosing groups of k from a pool of n.
    
    Parameters
    ----------
    n: int
        Number of items in the pool you are choosing from
    k: int
        Size of the groups you are choosing from the pool
        
    n!/(k!*(n-k)!)
    """
    return f(n)/f(k)/f(n-k)
    
def choose_AD_RD(AD_start, AD_end, RD_start, RD_end, AD_num, RD_num):
    """
    Parameters
    ----------
    AD_start: int
        Lowest axial diffusivity desired
    AD_end: int
        Highest axial diffusivity desired
    RD_start: int
        Lowest radial diffusivity desired
    RD_end: int
        Highest radial diffusivity desired
    AD_num: int
        Number of different axial diffusivities
    RD_num: int
        Number of different radial diffusivities
        
    Returns
    -------
    AD_combos: obj
        Unique axial diffusivity combinations
    RD_combos: obj
        Unique radial diffusivity combinations
    """
    
    AD_bag = np.linspace(AD_start, AD_end, num = AD_num)
    RD_bag = np.linspace(RD_start, RD_end, num = RD_num)

    AD_combos = list(itertools.combinations(AD_bag, 3))
    RD_combos = list(itertools.combinations(RD_bag, 3))
    
    return AD_combos, RD_combos
    
def predict_RD_AD(AD_start, AD_end, RD_start, RD_end, AD_num, RD_num, data, bvals, bvecs, mask):
    """
    Predicts vertices with different axial and radial diffusivities and finds them
    root-mean-square error (rmse) between the actual values and predicted values.
    
    Parameters
    ----------
    AD_start: int
        Lowest axial diffusivity desired
    AD_end: int
        Highest axial diffusivity desired
    RD_start: int
        Lowest radial diffusivity desired
    RD_end: int
        Highest radial diffusivity desired
    AD_num: int
        Number of different axial diffusivities
    RD_num: int
        Number of different radial diffusivities
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
        
    Returns
    -------
    rmse_b: 1 dimensional array
        The rmse from fitting to individual b values
    rmse_mb: 1 dimensional array
        The rmse from fitting to all the b values
    AD_order: list
        The groups of axial diffusivities in the order they were chosen
    RD_order: list
        The groups of radial diffusivities in the order they were chosen
    """
    AD_combos, RD_combos = choose_AD_RD(AD_start, AD_end, RD_start, RD_end, AD_num, RD_num)
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
    num_bvals = len(unique_b[1:])
    
    AD_order = []
    RD_order = []
    rmse_b = np.empty((np.sum(mask), nchoosek(AD_num, num_bvals)*nchoosek(RD_num, num_bvals)))
    rmse_mb = np.empty(rmse_b.shape)

    track = 0
    for AD_idx in np.arange(len(AD_combos)):
        for RD_idx in np.arange(len(RD_combos)):
            actual_b, predicted_b = predict_n(data, bvals, bvecs, mask, np.array(AD_combos[AD_idx]), np.array(RD_combos[RD_idx]), 10, 'bvals')
            actual, predicted = predict_n(data, bvals, bvecs, mask, np.array(AD_combos[AD_idx]), np.array(RD_combos[RD_idx]), 10, 'all')
            
            rmse_b[:, track] = np.sqrt(np.mean((actual_b - predicted_b)**2, -1))
            rmse_mb[:, track] = np.sqrt(np.mean((actual - predicted)**2, -1))
            
            track = track + 1
            
            AD_order.append(AD_combos[AD_idx])
            RD_order.append(RD_combos[RD_idx])
            
    return rmse_b, rmse_mb, AD_order, RD_order
    
def place_predict(file_names, mask_vox_num, expected_file_num, file_path = os.getcwd(), save = "No", file_vol = "No"):
    
    data_path = "/biac4/wandell/data/klchan13/100307/Diffusion/data"
    files = os.listdir(file_path)

    # Get file object
    data_file = nib.load(os.path.join(data_path, "data.nii.gz"))
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_registered.nii.gz"))

    # Get data and indices
    wm_data = wm_data_file.get_data()
    wm_idx = np.where(wm_data==1)

    # Get b values
    bvals = np.loadtxt(os.path.join(data_path, "bvals"))
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    
    # For producing the volumes as an output
    vol_list = []
    
    # Keep track of files in case there are any missing ones
    i_track = np.ones(expected_file_num)
    for fn in file_names:
        vol = ozu.nans((wm_data_file.shape + all_b_idx.shape))
        for f_idx in np.arange(len(files)):
            this_file = files[f_idx]
            if this_file[(len(this_file)-6):len(this_file)] == "nii.gz":
                sub_data = nib.load(os.path.join(file_path, this_file)).get_data()
                if this_file[0:len(fn)] == fn:
                    i = int(this_file.split(".")[0][len(fn):])
                    
                    low = i*mask_vox_num
                    high = np.min([(i+1) * mask_vox_num, int(np.sum(wm_data))])
                    
                    # Now set the mask:
                    mask = np.zeros(wm_data_file.shape)
                    mask[wm_idx[0][low:high], wm_idx[1][low:high], wm_idx[2][low:high]] = 1
                    
                    if file_vol is "No":
                        vol[np.where(mask)] = sub_data
                    elif file_vol is "Yes":
                        vol[np.where(mask)] = sub_data[np.where(mask)]
                        
                    i_track[i] = 0
        vol_list.append(vol)
        if save is "Yes":
            aff = data_file.get_affine()
            nib.Nifti1Image(vol, aff).to_filename("vol_%s.nii.gz"%fn)
            
    missing_files = np.squeeze(np.where(i_track))

    return missing_files, vol_list
