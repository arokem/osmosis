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
import osmosis.multi_bvals_empirical as sfm_mb_e
import osmosis.snr as snr

def partial_round(bvals, factor = 1000.):
    """
    Round only the values about equal to zero.
    
    Parameters
    ----------
    bvals: 1 dimensional array
        All b values
    factor: int
        Value to first divide the b values first before rounding
    
    Returns
    -------
    partially_rounded: 1 dimensional array
        B values with values ~ 0 replaced with 0.
    """
    partially_rounded = bvals
    for j in np.arange(len(bvals/factor)):
        if round(bvals[j]/factor) == 0:
            partially_rounded[j] = 0
            
    return partially_rounded
def create_combos(bvecs, bvals, data, these_b_inds,
                  these_b_inds_rm0, all_inc_0, vec_pool, num_choose, combo_num):
    """
    Helper function for cross-validation functions
    
    Parameters
    ----------
    bvecs: 2 dimensional array
        All the b vectors
    bvals: 1 dimensional array
        All b values
    data: 4 dimensional array
        Diffusion MRI data
    these_b_inds: 1 dimensional array
        Indices currently undergoing k-fold cross-validation
    these_b_inds_rm0: 1 dimensional array
        Indices currently undergoing k-fold cross-validation but with respect to
        the non-zero b values.
    all_inc_0: 1 dimensional array
        these_b_inds concatenated to the b = 0 indices
    vec_pool: 1 dimensional array
        Shuffled indices corresponding to each of the values in these_b_inds
    num_choose: int
        Number of b values to leave out at a time.
    combo_num: int
        Current fold of k-fold cross-validation
    
    Returns
    -------
    si: 1 dimensional array
        Sorted indices after removing a certain number of indices for k-fold
        cross-validation
    vec_combo: 1 dimensional array
        Indices of b values to leave out for the current fold
    vec_combo_rm0: 1 dimensional array
        Indices of b values to leave out for the current fold with respect to
        the non-zero b values
    vec_pool_inds: 1 dimensional array
        Shuffled indices to leave out during the current fold corresponding
        to certain values in these_b_inds
    these_bvecs: 2 dimensional array
        B vectors with a certain number of vectors from the fold removed
    these_bvals: 1 dimensional array
        B values with a certain number of b values from the fold removed
    this_data: 4 dimensional array
        Diffusion data with a certain number of directions from the fold removed
    these_inc0: 1 dimensional array
        Sorted indices to the directions to fit to
    """
    
    idx = list(these_b_inds_rm0)
    these_inc0 = list(all_inc_0)
    
    low = int((combo_num)*num_choose)
    high = np.min([int(combo_num*num_choose + num_choose), len(vec_pool)])
    
    vec_pool_inds = vec_pool[low:high]
    vec_combo = these_b_inds[vec_pool_inds]
    vec_combo_rm0 = these_b_inds_rm0[vec_pool_inds]
        
    # Remove the chosen indices from the rest of the indices
    for choice_idx in vec_pool_inds:
        these_inc0.remove(these_b_inds[choice_idx])
        idx.remove(these_b_inds_rm0[choice_idx])

    # Make the list back into an array
    these_inc0 = np.array(sorted(these_inc0))
    
    # Need to sort the indices first before indexing full model's
    # regressors
    si = sorted(idx)
    
    # Isolate the b vectors, b values, and data not including those
    # to be predicted
    these_bvecs = bvecs[:, these_inc0]
    these_bvals = bvals[these_inc0]
    this_data = data[:, :, :, these_inc0]
    
    return [si, vec_combo, vec_combo_rm0, vec_pool_inds, these_bvecs,
                                these_bvals, this_data, these_inc0]
    
def new_mean_combos(vec_pool_inds, data, bvals, bvecs, mask, bounds, b_inds,
                         these_b_inds = None, b_idx1 = None, b_idx2 = None):
    """
    Helper function for calculating a new mean from all b values and corresponding data
    
    Parameters
    ----------
    vec_pool_inds: 1 dimensional array
        Shuffled indices to leave out during the current fold corresponding
        to certain values in these_b_inds
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 2 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    bounds: list
        List containing tuples indicating the bounds for each parameter in
        the mean model function.
    b_inds: list
        List of indices corresponding to each b value
    these_b_inds: 1 dimensional array
        Indices currently undergoing k-fold cross-validation.  Used for single fODF
    b_idx1: int
        First index into b_inds
    b_idx2: int
        Index into b_inds.
    
    Returns
    -------
    sig_out: 1 dimensional array
        New means for each b value in each direction
    new_params: 2 dimensional array
        New parameters for the mean model at each voxel
    b_inds_ar[b_idx1]: 1 dimensional array
        Indices corresponding to the b values to fit to.  Used for multi fODF
    inds_arr: 1 dimensional array
        Indices corresponding to the b values to fit to.  Used for single fODF
    """    
    inds_list = list(np.arange(len(bvals)))
    
    # Remove combo indices from the indices from all the b values.
    for vc in vec_pool_inds:
        if b_idx1 is not None:
            inds_list.remove(b_inds[1:][b_idx1][vc])
        elif b_idx2 is not None:
            inds_list.remove(b_inds[1:][b_idx2][vc])
        elif these_b_inds is not None:
            inds_list.remove(these_b_inds[vc])

    inds_arr = np.array(inds_list)
    
    # New data with removed indices.
    fit_all_data = data[..., inds_arr]
    fit_all_bvecs = bvecs[:, inds_arr]
    fit_all_bvals = bvals[inds_arr]
    
    # Now put this into SFM multi_b class in order to grab the calculated mean
    mod = sfm_mb.SparseDeconvolutionModelMultiB(fit_all_data, fit_all_bvecs, fit_all_bvals,
                                                mask = mask, bounds = bounds,
                                                params_file = "temp")
    _, b_inds_ar, _, _ = separate_bvals(fit_all_bvals, mode = "remove0")
    
    # Get the new means at each direction and the new parameters at each voxel
    sig_out, new_params = mod.fit_flat_rel_sig_avg
    
    if b_idx1 != None:
        _, b_inds_ar, _, _ = separate_bvals(fit_all_bvals, mode = "remove0")
        return sig_out, new_params, b_inds_ar[b_idx1]
    else:
        for bi0 in b_inds[0]:
            # Remove the b = 0 indices so we can obtain the indices to the
            # non diffusion weighted directions
            inds_list.remove(bi0)
        inds_arr = np.array(inds_list)
        return sig_out, new_params, inds_arr

def _predict_across_b(mod_obj, vec_combo, vec_pool_inds, bvecs, bvals,
                        b_inds, b_across, new_params = None):
    """
    Helper predict function for predicting across b values.
    
    Parameters
    ----------
    mod_obj: object
        Reduced model object
    vec_combo: 1 dimensional array
        Indices of b values to leave out for the current fold
    vec_pool_inds: 1 dimensional array
        Shuffled indices to leave out during the current fold corresponding
        to certain values in these_b_inds
    bvecs: 2 dimensional array
        All the b vectors
    bvals: 1 dimensional array
        All b values
    b_inds: list
        List of indices corresponding to each b value
    b_across: int
        Index into b_inds corresponding to the b values to predict
    new_params: 2 dimensional array
        New parameters for the mean model
    
    Returns
    -------
    predicted_across: 2 dimensional array
        Predicted values at each direction for predicting across b values
    predicted_to: 2 dimensional array
        Predicted values at each direction for predicting to the same b values
    """
    predicted_across = mod_obj.predict(bvecs[:, b_inds[1:][b_across][vec_pool_inds]],
                                   bvals[b_inds[1:][b_across][vec_pool_inds]], 
                                   new_params = new_params)[mod.mask]
    predicted_to = mod_obj.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                   new_params = new_params)[mod.mask]
                            
    predicted_across, predicted_to
    
def _preload_regressors(si, full_mod_obj, mod_obj, over_sample, mean, fODF_mode):
    """
    Helper function for grabbing reduced versions of preloaded regressors.
    
    Parameters
    ----------
    si: 1 dimensional array
        Sorted indices after removing a certain number of indices for k-fold
        cross-validation
    full_mod_obj: object
        Model object including all direction
    mod_obj: object
        Model object including a reduced number of directions
    mean: str
        String indicating what kind of mean to use.
    fODF_mode: str
        String indicating how many fODFs are to be made.
        
    Returns
    -------
    fit_to: 2 dimensional array
        Values to fit to at each voxel and at each direction
    tensor_regressor: 2 dimensional array
        Non-demeaned design matrix for fitting
    fit_to_demeaned: 2 dimensional array
        Values to fit to and demeaned by some kind of mean at each voxel
        and at each direction
    fit_to_means: 1 dimensional array
        Means at each direction with which the fit_to values and design_matrix
        were demeaned
    design_matrix: 2 dimensional array
        Demeaned design matrix for fitting
    """
    fit_to = full_mod_obj.regressors[0][:, si]
    if over_sample is None:
        # Take the reduced vectors from both the b vectors and rotational vectors
        tensor_regressor = full_mod_obj.regressors[1][:, si][si, :]
        if mean is "MD":
            design_matrix = full_mod_obj.regressors[4][:, si][si, :]
    else:
        # If over or under sampling, the rotational vectors are not equal to
        # the original b vectors so you shouldn't take the reduced amount
        # of rotational vectors.
        tensor_regressor = full_mod_obj.regressors[1][si, :]
        if mean is "MD":
            design_matrix = full_mod_obj.regressors[4][si, :]
    
    fit_to_demeaned = full_mod_obj.regressors[2][:, si]
    fit_to_means = full_mod_obj.regressors[3][:, si]
    
    if mean is "mean_model":
        # Want the average signal from mean model fit to just the reduced data
        sig_out,_ = mod_obj.fit_flat_rel_sig_avg
        return fit_to, tensor_regressor, fit_to_demeaned, sig_out
    elif mean is "MD":
        return fit_to, tensor_regressor, fit_to_demeaned, fit_to_means, design_matrix
   
def _kfold_xval_setup(bvals, mask):
    """
    Heler function to help set up any separation of b values and initial reallocating
    
    Parameters
    ----------
    bvals: 1 dimensional array
        All b values
    mask: 3 dimensional array
        Brain mask of the data
    
    Returns
    -------
    b_inds: list
        List of indices corresponding to each b value
    unique_b: 1 dimensional array
        All the unique b values including zero
    b_inds_rm0: list
        List of indices corresponding to each non-zero b value
    all_b_idx: 1 dimensional array
        Indices corresponding to all non-zero b values
    all_b_idx_rm0: 1 dimensional array
        Indicecs corresponding to all b values with respect to the non-zero b values
    """
    # Separating b values and grabbing indices
    bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)    
    _, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = separate_bvals(bvals,
                                                                    mode = 'remove0')
    # Grab the indices where b is not equal to 0
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    all_b_idx_rm0 = np.arange(len(all_b_idx))
    
    # Preallocate predict output
    predicted = np.empty((int(np.sum(mask)), len(all_b_idx)))
    
    return b_inds, unique_b, b_inds_rm0, all_b_idx, all_b_idx_rm0, predicted

def kfold_xval(data, bvals, bvecs, mask, ad, rd, n, fODF_mode,
              mean = "mean_model", sph_cc = False, fit_method = None,
              b_idx1 = None, b_idx2 = None, over_sample=None,
              bounds = None, solver=None):
    """
    Does k-fold cross-validation leaving out a certain percentage of the vertices
    out at a time.  This function can be used for 7 different variations of
    k-fold cross validation.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 2 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    n: int
        Integer indicating the percent of vertices that you want to predict
    fODF_mode: str
        'single': if fitting to all b values to create a single fODF
        'multi': if fitting to individual b values to create multiple fODFs
    mean: str
        'mean_model': Demean the design matrix and signals using one of the
        mean models from the mean_diffusivity_models module
        'empirical': Demean the design matrix and signals using the empirical mean
        'MD': Demean the design matrix and signals using the mean diffusivity
    sph_cc: Boolean
        True: Does not predict the left out vertices but instead computes the
        spherical cross-correlation between different model parameters
        False: Performs k-fold cross-validation.
    fit_method: str
        "WLS": Weighted least squares.  Default is the least squares method
        specified by the solver.
    b_idx1: int
        First index into the unique b values (not including zero) to reduce the
        data to for predicting across b values
    b_idx2: int
        Second index into the unique b values (not including zero) to reduce the
        data to for predicting across b values
    over_sample: int
        Number of rotational vectors to oversample or undersample to if you
        don't want to use the b vectors as the rotational vectors.
    bounds: list of tuples
        Bounds on the parameters for fitting the mean model
    solver: str
        Solver to be used in multi_bvals module for fitting the SFM.
        
    Returns
    -------
    actual: 2 dimensional array
        Actual signals for the predicted vertices
    predicted: 2 dimensional array 
        Predicted signals for the vertices left out of the fit
    cc_list: list
        List of spherical correlation coefficients between unique pairs of
        model parameters
    """ 
    t1 = time.time()
    [b_inds, unique_b, b_inds_rm0,
    all_b_idx, all_b_idx_rm0, predicted] = _kfold_xval_setup(bvals, mask)
    
    if (mean is "empirical") & (fODF_mode is "single"):
        # Use an entirely different module if demeaning by the empirical
        # mean to create a single fODF for each voxel
        module = sfm_mb_e
    else:
        module = sfm_mb
    
    # Generate the regressors in the full model from which we choose the regressors in
    # the reduced model from.  This is so you won't have to recalculate the regressors
    # each time you do cross-validation.     
    full_mod = module.SparseDeconvolutionModelMultiB(data, bvecs, bvals, mask = mask,
                                                    axial_diffusivity = ad,
                                                    radial_diffusivity = rd,
                                                    over_sample = over_sample,
                                                    bounds = bounds, solver = solver,
                                                    fit_method = fit_method,
                                                    mean = mean, params_file = "temp")
    if sph_cc is True:
        cc_list = []
    
    if fODF_mode is "single":
        indices = np.array([0])
    elif fODF_mode is "multi":
        if b_idx2 is None:
            indices = np.arange(len(unique_b[1:]))
        else:
            # Predict across b values if given a second b_idx (b_idx2)
            indices = np.array([b_idx1, b_idx2])
            # Initialize 4 different outputs - one for each prediction.
            predicted11 = np.empty((int(np.sum(mask)),) + (len(b_inds[1]),))
            predicted12 = np.empty(predicted11.shape)
            predicted21 = np.empty(predicted11.shape)
            predicted22 = np.empty(predicted11.shape)
    
    mp_list = []

    for bi in indices:
        if fODF_mode is "single":
            all_inc_0 = np.arange(len(bvals))
            # Indices are locations of all non-b = 0
            these_b_inds = all_b_idx
            these_b_inds_rm0 = all_b_idx_rm0
        elif fODF_mode is "multi":
            # Indices of data with a particular b value
            all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][bi]))
            these_b_inds = b_inds[1:][bi]
            these_b_inds_rm0 = b_inds_rm0[bi]
        vec_pool = np.arange(len(these_b_inds))
        
        # Need to choose random indices so shuffle them.
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
            # Create the combinations of directions to leave out at a time and remove them
            # from the original data for fitting purposes.
            (si, vec_combo, vec_combo_rm0,
            vec_pool_inds, these_bvecs, these_bvals,
            this_data, these_inc0) = create_combos(bvecs, bvals, data, these_b_inds,
                                                    these_b_inds_rm0, all_inc_0, vec_pool,
                                                    num_choose, combo_num)                
            # Initial a model object with reduced data (not including the chosen combinations)    
            mod = module.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                         mask = mask, axial_diffusivity = ad,
                                                         radial_diffusivity = rd,
                                                         over_sample = over_sample,
                                                         bounds = bounds, solver = solver,
                                                         fit_method = fit_method,
                                                         mean = mean, params_file = "temp")
            #if bi == 0:
                #mp_list.append(np.zeros(len(indices)*len(these_b_inds)))
                
            if (mean is "mean_model") & (fODF_mode is not "single"):
                # Get the parameters from fitting a mean model to all b values not including
                # the ones left out.  The "single" b model already fetches a mean according fit
                # to all b values
                if (fODF_mode is "multi") & (b_idx2 == None):
                    b_mean1 = bi
                elif b_idx2 is not None:
                    b_mean1 = b_idx1
                # Fit a new mean model to all data except the chosen combinations.
                sig_out, new_params, b_inds_ar = new_mean_combos(vec_pool_inds, data, bvals, bvecs,
                                           mask, bounds, b_inds, b_idx1 = b_mean1, b_idx2 = b_idx2)
                if (fODF_mode is "multi") & (b_idx2 == None):
                    # Replace the relative signal average of the model object with one calculated.
                    mod.fit_flat_rel_sig_avg = [sig_out[:, b_inds_ar], new_params]
            else:
                new_params = None

            # Grab regressors from full model's preloaded regressors.  This only works if
            # not predicting across b values.
            if b_idx2 == None: #(mean is not "empirical") & (b_idx2 == None)
                if mean is "MD":
                    full_mod.tensor_model.mean_diffusivity = mod.tensor_model.mean_diffusivity
                    
                if ((mean is "MD") & (fODF_mode is "multi")) | (mean is "empirical"):
                    mod.regressors
                else:
                    mod.regressors = _preload_regressors(si, full_mod, mod,
                                                        over_sample, mean, fODF_mode)
            if sph_cc is False:
                if b_idx2 != None:
                    # Since we're using a separate output for each prediction, and not indexing
                    # into one big array, use the indices starting from 0:
                    vec_combo_rm0 = vec_pool_inds
                    if bi == b_idx1:
                        [predicted12[:, vec_pool_inds],
                        predicted11[:, vec_combo_rm0]] = _predict_across_b(mod, vec_combo,
                            vec_pool_inds, bvecs, bvals, b_inds, b_idx2, new_params = new_params)
                    else:
                        [predicted21[:, vec_pool_inds],
                        predicted22[:, vec_combo_rm0]] = _predict_across_b(mod, vec_combo,
                            vec_pool_inds, bvecs, bvals, b_inds, b_idx2, new_params = new_params)
                else:
                    predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                                                new_params = new_params)[mod.mask]
            else:
                mp_list.append(mod.model_params[mod.mask])
        if sph_cc is True:
            cc_arr = kfold_xval_sph_cc(mp_list, mask, mod.rot_vecs)
            cc_list.append(cc_arr)
            
    t2 = time.time()
    print "This program took %4.2f minutes to run"%((t2 - t1)/60)
    
    if b_idx2 != None:
        actual1 = data[mod.mask][:, b_inds[1:][b_idx1]] # Actual values for b_idx1
        actual2 = data[mod.mask][:, b_inds[1:][b_idx2]] # Actual values for b_idx2
        return actual1, actual2, predicted11, predicted12, predicted22, predicted21
    elif sph_cc is True:
        return cc_list
    else:
        actual = data[mod.mask][:, all_b_idx]
        return actual, predicted
        
def kfold_xval_sph_cc(mp_list, mask, rot_vecs):
    """
    Helper function that finds the spherical cross-correlation between the different
    model params generated by k-fold cross-validation.
    
    Parameters
    ----------
    mp_list: list
        List containing all the model parameters from k-fold cross-validation
    mask: 3 dimensional array
        Brain mask of the data
    rot_vecs: 2 dimensional array
        All the rotational vectors to which the weights in the model parameters
        correspond to
        
    Returns
    -------
    cc_arr: 2 dimensional array
        An array consisting of the spherical cross-correlations between each of the
        combinations of model parameters at each voxel
    """
    # Calculate the number of combinations that are created
    num_combos = nchoosek(len(mp_list),2)
    
    # Preallocate an array to include one for each combination per voxel
    cc_arr = np.zeros((num_combos, int(np.sum(mask))))
    mp_count = 0
    for mp_inds in itertools.combinations(np.arange(len(mp_list)), 2):
        for vox in np.arange(int(np.sum(mask))):
            # Mirror the data and the rotational vectors
            mp_mirr1 = np.concatenate((mp_list[mp_inds[0]][vox], mp_list[mp_inds[0]][vox]), -1)
            mp_mirr2 = np.concatenate((mp_list[mp_inds[1]][vox], mp_list[mp_inds[1]][vox]), -1)
            bvecs_mirr = np.squeeze(np.concatenate((rot_vecs,-1*rot_vecs), -1)).T
            deg, cc = ozu.sph_cc(mp_mirr1, mp_mirr2, bvecs_mirr)
            if all(~np.isfinite(cc)):
                cc_arr[mp_count][vox] = nan
            else:
                cc_arr[mp_count][vox] = np.max(cc[np.isfinite(cc)]) # Maximum of the non-nan values
            
        mp_count = mp_count + 1
    
    return cc_arr

def predict_grid(data, bvals, bvecs, mask, ad, rd, n, over_sample = None, solver = None,
                                                      fit_method = None, bounds = None):
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
    fODF_mode: str
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
    
    [b_inds, unique_b, b_inds_rm0,
    all_b_idx, all_b_idx_rm0, predicted] = _kfold_xval_setup(bvals, mask)
    actual = np.empty(predicted.shape)
    
    # Initialize a list for each group of directions
    group_list = []
    for ub in np.arange(1, len(unique_b)):
        these_inds = np.arange(len(b_inds[ub]))
        # Shuffle the indices in each b value for making groups
        np.random.shuffle(these_inds)
        group_list.append(these_inds)
    # Initialize a full model object with all directions for preloading regressors
    full_mod = sfm_mb.SparseDeconvolutionModelMultiB(data, bvecs, bvals, mask = mask,
                                                axial_diffusivity = ad,
                                                radial_diffusivity = rd,
                                                over_sample = over_sample,
                                                bounds = bounds, solver = solver,
                                                fit_method = fit_method,
                                                params_file = "temp")
    for bi in np.arange(3):                                                    
        these_b_inds = np.array([]).astype(int)
        these_b_inds_rm0 = np.array([]).astype(int)
        for ub2 in np.arange(len(group_list)):
            num_lo = int(len(group_list[ub2])/3.) # Number left out at each fold
            this_group = group_list[ub2][bi*num_lo:(bi*num_lo + num_lo)]
            these_b_inds = np.concatenate((these_b_inds, b_inds[ub2+1][this_group]))
            these_b_inds_rm0 = np.concatenate((these_b_inds_rm0, b_inds_rm0[ub2][this_group]))

        all_inc_0 = np.concatenate((b_inds[0], these_b_inds))           
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
             this_data, these_inc0) = create_combos(bvecs, bvals, data, these_b_inds,
                                                        these_b_inds_rm0, all_inc_0,
                                                        vec_pool, num_choose, combo_num)
            
            mod = sfm_mb.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                        mask = mask, params_file = "temp",
                                                        axial_diffusivity = ad,
                                                        radial_diffusivity = rd,
                                                        bounds = bounds,
                                                        over_sample = over_sample,
                                                        solver = solver)
            sig_out, new_params, inds_arr = new_mean_combos(vec_pool_inds, data, bvals, bvecs,
                                             mask, bounds, b_inds, these_b_inds = these_b_inds)
            mod.fit_flat_rel_sig_avg = [sig_out[:, inds_arr], new_params]
            mod.regressors = _preload_regressors(si, full_mod, mod, over_sample)
            predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                                      new_params = new_params)[mod.mask]
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

    

def kfold_xval_gen(model_class, data, bvecs, bvals, k, mask = None, fODF_mode = "single", **kwargs):
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
    fODF_mode: str
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
    
    # Generate the regressors in the full model from which we choose the regressors in
    # the reduced model from.
    if fODF_mode is "single": 
        indices = np.array([0])
    elif fODF_mode is "multi":
        indices = np.arange(len(unique_b[1:]))
    
    for bi in indices:
        if fODF_mode is "single":
            all_inc_0 = np.arange(len(rounded_bvals))
            these_b_inds = all_b_idx
            these_b_inds_rm0 = all_b_idx_rm0
        elif fODF_mode is "multi":
            all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][bi]))
            these_b_inds = b_inds[1:][bi]
            these_b_inds_rm0 = b_inds_rm0[bi]
        bvals_pool = partial_round(bvals)
        vec_pool = np.arange(len(these_b_inds))
        
        # Need to choose random indices so shuffle them!
        np.random.shuffle(vec_pool)
        
        # How many of the indices are you going to leave out at a time?
        num_choose = (k/100.)*len(these_b_inds)
 
        for combo_num in np.arange(np.floor(100./k)):
            (si, vec_combo, vec_combo_rm0,
            vec_pool_inds, these_bvecs, these_bvals,
            this_data, these_inc0) = create_combos(bvecs, bvals_pool, data, these_b_inds,
                                                   these_b_inds_rm0, all_inc_0, vec_pool,
                                                                  num_choose, combo_num)
            
            mod = model_class(this_data, these_bvecs, these_bvals,
                                  mask = mask, params_file = "temp", **kwargs)
            # If the number of inputs is greater than two, then the b values are needed                                     
            if len(inspect.getargspec(mod.predict)[0]) > 2:
                predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo],
                                              bvals[vec_combo])[mod.mask]
            # If the number of inputs is equal to two, then the b values are not needed
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
        # If you want to predict using k-fold cross-validation:
        [actual1, actual2, predicted11,
        predicted12, predicted22, predicted21] = predict_n(data, bvals, bvecs, mask, ad, rd, n,
                                                           fODF_mode = "multi", b_idx1 = b_idx1,
                                                           b_idx2 = b_idx2, new_mean = "Yes",
                                                           solver = solver)
    else:
        # If you want to predict normally, not using k-fold cross-validation
        mod = sfm_mb.SparseDeconvolutionModelMultiB(data[:,:,:,all_inc_0],
                                                    bvecs[:,all_inc_0], bvals[all_inc_0],
                                                    mask = mask, axial_diffusivity = ad,
                                                    radial_diffusivity = rd, solver = solver,
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
            # Predict with different AD, RD values.
            actual_b, predicted_b = predict_n(data, bvals, bvecs, mask, np.array(AD_combos[AD_idx]), np.array(RD_combos[RD_idx]), 10, 'bvals')
            actual, predicted = predict_n(data, bvals, bvecs, mask, np.array(AD_combos[AD_idx]), np.array(RD_combos[RD_idx]), 10, 'all')
            
            # Calculate RMSE values.
            rmse_b[:, track] = np.sqrt(np.mean((actual_b - predicted_b)**2, -1))
            rmse_mb[:, track] = np.sqrt(np.mean((actual - predicted)**2, -1))
            
            track = track + 1
            
            AD_order.append(AD_combos[AD_idx])
            RD_order.append(RD_combos[RD_idx])
            
    return rmse_b, rmse_mb, AD_order, RD_order