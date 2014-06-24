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
import cv

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
    
def new_mean_combos(vec_pool_inds, data, bvals, bvecs, mask, b_inds, bounds="preset",
                    mean_mod_func = "bi_exp_rs", these_b_inds=None, b_idx1=None, b_idx2=None):
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
    mod = sfm.SparseDeconvolutionModelMultiB(fit_all_data, fit_all_bvecs, fit_all_bvals,
                                                mask = mask, bounds=bounds,
                                                mean_mod_func = mean_mod_func,
                                                params_file = "temp")
    _, b_inds_ar, _, _ = separate_bvals(fit_all_bvals, mode = "remove0")
    
    # Get the new means at each direction and the new parameters at each voxel
    sig_out, new_params = mod.fit_flat_rel_sig_avg

    if b_idx1 is not None:
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
                                   new_params = new_params)[mod_obj.mask]
    predicted_to = mod_obj.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                   new_params = new_params)[mod_obj.mask]

    predicted_across, predicted_to
    
def _preload_regressors(si, full_mod_obj, mod_obj, over_sample, mean):
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
        if mean == "MD":
            design_matrix = full_mod_obj.regressors[4][:, si][si, :]
    else:
        # If over or under sampling, the rotational vectors are not equal to
        # the original b vectors so you shouldn't take the reduced amount
        # of rotational vectors.
        tensor_regressor = full_mod_obj.regressors[1][si, :]
        if mean == "MD":
            design_matrix = full_mod_obj.regressors[4][si, :]
    
    fit_to_demeaned = full_mod_obj.regressors[2][:, si]
    fit_to_means = full_mod_obj.regressors[3][:, si]
    
    if mean == "mean_model":
        # Want the average signal from mean model fit to just the reduced data
        sig_out,_ = mod_obj.fit_flat_rel_sig_avg
        return fit_to, tensor_regressor, fit_to_demeaned, sig_out
    elif mean == "MD":
        return fit_to, tensor_regressor, fit_to_demeaned, fit_to_means, design_matrix
   
def _kfold_xval_setup(bvals, mask):
    """
    Helper function to help set up any separation of b values and initial reallocating
    
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
    
def _aggregate_fODFs(all_mp_list, all_mp_rot_vecs_list, unique_b, precision, start_fODF_mode):
    """
    Changes the arrangement of the model parameters and rotational vectors list so that all the multi fODF
    parameters are together in one list slot.
    """
    # Add the single fODFs' model parameters and rotational vectors first
    out_mp_list = []
    out_rot_vecs_list = []
    indices = np.array([0])
    if precision != "emd_multi_combine":
        if start_fODF_mode == "both_ms":
            # single fODF model params are last on the list for both_ms
            out_mp_list = [all_mp_list[len(all_mp_list)-1]]
            out_rot_vecs_list = [all_mp_rot_vecs_list[len(all_mp_list)-1]]
        elif start_fODF_mode == "both_m":
            indices = np.array([0,len(all_mp_list)/2])
    
    # Start some new temporary lists to add the multi fODF parameters to before appending them to the
    # output list.
    for ii in indices:
        temp_mp_list = []
        temp_rot_vecs_list = []
        for mp_idx in np.arange(len(all_mp_list[0])):
            mp_arr = all_mp_list[ii][mp_idx]
            rot_vecs_arr = all_mp_rot_vecs_list[ii][mp_idx]
            
            # mp_list differs depending on the fODF mode
            if ((precision == "emd_multi_combine") |
                (start_fODF_mode == "both_ms")):
                start = 1
                end = len(unique_b[1:])
            elif start_fODF_mode == "both_m":
                start = ii + 1
                end = ii + len(unique_b[1:])
             
            for b_idx in np.arange(start, end):
                mp_arr = np.concatenate((mp_arr,
                                         all_mp_list[b_idx][mp_idx]),-1)
                rot_vecs_arr = np.concatenate((rot_vecs_arr,
                                               all_mp_rot_vecs_list[b_idx][
                                               mp_idx]),-1)
            
            temp_mp_list.append(mp_arr)
            temp_rot_vecs_list.append(rot_vecs_arr)

        out_mp_list.append(temp_mp_list)
        out_rot_vecs_list.append(temp_rot_vecs_list)

    return np.squeeze(out_mp_list), np.squeeze(out_rot_vecs_list)
       
def _calc_precision(mp1, mp2, rot_vecs1, rot_vecs2, idx1, idx2, mp_count, vox, p_arr, precision_type):
    """
    Does the actual precision calculation.
    """
    if precision_type == "sph_cc":
        # Mirror the data and the rotational vectors
        mp_mirr1 = np.concatenate((mp1, mp1), -1)
        mp_mirr2 = np.concatenate((mp2, mp2), -1)
        bvecs_mirr1 = np.squeeze(np.concatenate((rot_vecs1,-1*rot_vecs1), -1)).T
        bvecs_mirr2 = np.squeeze(np.concatenate((rot_vecs2,-1*rot_vecs2), -1)).T
        deg, p = ozu.sph_cc(mp_mirr1, mp_mirr2, bvecs_mirr1, vertices2=bvecs_mirr2)
        
        if all(~np.isfinite(cc)):
            p_arr[mp_count][vox] = np.nan
        else:
            p_arr[mp_count][vox] = np.max(cc[np.isfinite(cc)]) # Maximum of the non-nan values
    
    elif (precision_type == "emd") | (precision_type == "emd_multi_combine"):
        if ((len(np.where(mp1)[0]) != 0) & (len(np.where(mp2)[0]) != 0)):
            emd = fODF_EMD(mp1[idx1], mp2[idx2], rot_vecs1[:, idx1], rot_vecs2[:, idx2])
            p_arr[mp_count][vox] = emd
        else:
            p_arr[mp_count][vox] = np.nan

    return p_arr
    
def kfold_xval(data, bvals, bvecs, mask, ad, rd, n, fODF_mode,
               mean_mod_func = "bi_exp_rs", mean = "mean_model",
               mean_mix = None, precision = False, fit_method = None,
               b_idx1 = None, b_idx2 = None, over_sample=None,
               bounds = "preset", solver=None, viz = False):
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
    precision: Boolean <--------CHANGE ME
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
    
    # Generate the regressors in the full model from which we choose the regressors in
    # the reduced model from.  This is so you won't have to recalculate the regressors
    # each time you do cross-validation.     
    full_mod = sfm.SparseDeconvolutionModelMultiB(data, bvecs, bvals, mask = mask,
                                                    axial_diffusivity = ad,
                                                    radial_diffusivity = rd,
                                                    over_sample = over_sample,
                                                    bounds = bounds, solver = solver,
                                                    fit_method = fit_method,
                                                    mean_mix = mean_mix,
                                                    mean_mod_func = mean_mod_func,
                                                    mean = mean, params_file = "temp")
    if precision is not False:
        p_list = []
        if precision == "emd_multi_combine":
            all_mp_list = []
            all_mp_rot_vecs_list = []
    
    start_fODF_mode = "None"
    if fODF_mode == "single":
        indices = np.array([0])
    elif fODF_mode == "multi":
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
    elif fODF_mode == "both_ms":
        start_fODF_mode = "both_ms"
        indices = np.arange(len(unique_b[1:])+1)
        all_mp_list = []
        all_mp_rot_vecs_list = []
    elif fODF_mode == "both_s":
        start_fODF_mode = "both_s"
        indices = np.array([0,0])
        all_mp_list = []
        all_mp_rot_vecs_list = []
    elif fODF_mode == "both_m":
        start_fODF_mode = "both_m"
        indices = np.concatenate((np.arange(len(unique_b[1:])),
                                 np.arange(len(unique_b[1:]))))
        all_mp_list = []
        all_mp_rot_vecs_list = []
    
    count = 0
    for bi_idx, bi in enumerate(indices):
        mp_list = []
        mp_rot_vecs_list = []
        
        if (start_fODF_mode == "both_m") & (bi_idx == 0):
            mean_mod_func = "bi_exp_rs"
        elif (start_fODF_mode == "both_m") & (bi_idx == len(indices)/2):
            mean_mod_func = "single_exp_rs"
        elif (start_fODF_mode == "both_s") & (bi_idx == 0):
            mean_mod_func = "bi_exp_rs"
        elif (start_fODF_mode == "both_s") & (bi_idx == 1):
            mean_mod_func = "single_exp_rs"
            
        if (fODF_mode == "single") | ((start_fODF_mode == "both_ms") &
           (bi == len(unique_b[1:]))) | (start_fODF_mode == "both_s"):
            fODF_mode = "single"
            all_inc_0 = np.arange(len(bvals))
            # Indices are locations of all non-b = 0
            these_b_inds = all_b_idx
            these_b_inds_rm0 = all_b_idx_rm0
        elif (fODF_mode == "multi") | ((start_fODF_mode == "both_ms") &
             (bi != len(unique_b[1:]))) | (start_fODF_mode == "both_m"):
            fODF_mode = "multi"
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
            this_data, these_inc0) = ozu.create_combos(bvecs, bvals, data, these_b_inds,
                                                    these_b_inds_rm0, all_inc_0, vec_pool,
                                                    num_choose, combo_num)                
            # Initial a model object with reduced data (not including the chosen combinations)    
            mod = sfm.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                         mask = mask, axial_diffusivity = ad,
                                                         radial_diffusivity = rd,
                                                         over_sample = over_sample,
                                                         bounds = bounds, solver = solver,
                                                         fit_method = fit_method,
                                                         mean_mix = mean_mix,
                                                         mean_mod_func = mean_mod_func,
                                                         mean = mean, params_file = "temp")
                
            if (mean == "mean_model") & (fODF_mode != "single"):
                # Get the parameters from fitting a mean model to all b values not including
                # the ones left out.  The "single" b model already fetches a mean according fit
                # to all b values
                if (fODF_mode == "multi") & (b_idx2 == None):
                    b_mean1 = bi
                elif b_idx2 is not None:
                    b_mean1 = b_idx1
                    
                # Fit a new mean model to all data except the chosen combinations.
                sig_out, new_params, b_inds_ar = new_mean_combos(vec_pool_inds, data, bvals, bvecs,
                                        mask, b_inds, bounds=bounds, mean_mod_func = mean_mod_func,
                                        b_idx1=b_mean1, b_idx2=b_idx2)
                
                if (fODF_mode == "multi") & (b_idx2 == None):
                    # Replace the relative signal average of the model object with one calculated.
                    mod.fit_flat_rel_sig_avg = [sig_out[:, b_inds_ar], new_params]
            else:
                new_params = None

            # Grab regressors from full model's preloaded regressors.  This only works if
            # not predicting across b values.
            if b_idx2 == None:
                if mean == "MD":
                    full_mod.tensor_model.mean_diffusivity = mod.tensor_model.mean_diffusivity
                    
                if ((mean == "MD") & (fODF_mode == "multi")):
                    mod.regressors
                elif (mean == "empirical") | (mean_mix == "mm_emp"): #Use empirical regressors but mean model
                    mod.empirical_regressors
                else:
                    mod.regressors = _preload_regressors(si, full_mod, mod,
                                                        over_sample, mean)
            if precision is False:
                if b_idx2 != None:
                    # Since we're using a separate output for each prediction, and not indexing
                    # into one big array, use the indices starting from 0:
                    vec_combo_rm0 = vec_pool_inds
                    if bi == b_idx1:
                        predicted12[:, vec_pool_inds] = mod.predict(bvecs[:,
                        b_inds[1:][b_idx2][vec_pool_inds]], bvals[b_inds[1:][b_idx2][
                        vec_pool_inds]], new_params = new_params)[mod.mask]
                        predicted11[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo],
                                    bvals[vec_combo], new_params = new_params)[mod.mask]
                    else:
                        predicted21[:, vec_pool_inds] = mod.predict(bvecs[:,
                        b_inds[1:][b_idx1][vec_pool_inds]], bvals[b_inds[1:][b_idx1][
                        vec_pool_inds]], new_params = new_params)[mod.mask]
                        predicted22[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo],
                                    bvals[vec_combo], new_params = new_params)[mod.mask]
                else:
                    this_pred = mod.predict(bvecs[:, vec_combo], bvals[vec_combo],
                                                new_params = new_params)[mod.mask]
                    if count == 0:
                        predicted = np.zeros((np.sum(mod.mask), predicted.shape[1]))
                        count = count + 1
                    predicted[:, vec_combo_rm0] = this_pred
            else:
                #Save both the model params and their corresponding rotational vectors for precision function
                mp_list.append(mod.model_params[mod.mask])
                mp_rot_vecs_list.append(mod.rot_vecs)
                
        if (precision is not False) & (precision != "emd_multi_combine") & (start_fODF_mode[:4] != "both"):
            p_arr = kfold_xval_precision(mp_list, mod.mask, mp_rot_vecs_list, precision, start_fODF_mode)
            p_list.append(p_arr)

        if (start_fODF_mode[:4] == "both") | (precision == "emd_multi_combine"):
            all_mp_list.append(mp_list)
            all_mp_rot_vecs_list.append(mp_rot_vecs_list)

    if ((start_fODF_mode[:4] == "both") & (start_fODF_mode != "both_s") |
                                      (precision == "emd_multi_combine")):
        # Call to function to aggregate multi fODFs
        [mp_list, mp_rot_vecs_list] = _aggregate_fODFs(all_mp_list, all_mp_rot_vecs_list,
                                                       unique_b, precision, start_fODF_mode)
        p_arr = kfold_xval_precision(mp_list, mod.mask, mp_rot_vecs_list, precision, start_fODF_mode)
        p_list.append(p_arr)
    elif start_fODF_mode == "both_s":
        p_arr = kfold_xval_precision(all_mp_list, mod.mask, all_mp_rot_vecs_list, precision, start_fODF_mode)
        p_list.append(p_arr)
        
    t2 = time.time()
    print "This program took %4.2f minutes to run"%((t2 - t1)/60)
    
    if b_idx2 != None:
        actual1 = data[mod.mask][:, b_inds[1:][b_idx1]] # Actual values for b_idx1
        actual2 = data[mod.mask][:, b_inds[1:][b_idx2]] # Actual values for b_idx2
        return actual1, actual2, predicted11, predicted12, predicted22, predicted21
    elif precision is not False:
        if viz == True:
            return p_list, mp_list, mp_rot_vecs_list
        else:
            return p_list
    else:
        actual = data[mod.mask][:, all_b_idx]
        return actual, predicted
        
def kfold_xval_precision(mp_list, mask, rot_vecs_list, precision_type, start_fODF_mode):
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
    ***SOMETHING ABOUT PRECISION TYPE***
        
    Returns
    -------
    cc_arr: 2 dimensional array
        An array consisting of the spherical cross-correlations between each of the
        combinations of model parameters at each voxel
    """
    
    if start_fODF_mode == "None":
        itr = itertools.combinations(np.arange(len(mp_list)), 2)
    elif start_fODF_mode[:4] == "both":
        mp_list1_inds = list(np.arange(len(mp_list[0])))
        mp_list2_inds = list(np.arange(len(mp_list[1])))
        itr = itertools.product(*[mp_list1_inds, mp_list2_inds])
        
    # Calculate the number of combinations that are created
    if start_fODF_mode == "None":
        num_combos = nchoosek(len(mp_list),2)
    elif start_fODF_mode[:4] == "both":
        num_combos = len(mp_list1_inds)*len(mp_list2_inds)
    
    # Preallocate an array to include one for each combination per voxel
    p_arr = np.zeros((num_combos, int(np.sum(mask))))
    mp_count = 0
        
    for mp_inds in itr:
        for vox in np.arange(int(np.sum(mask))):
            if start_fODF_mode == "None":
                mp1 = mp_list[mp_inds[0]][vox]
                mp2 = mp_list[mp_inds[1]][vox]
                rot_vecs1 = rot_vecs_list[mp_inds[0]]
                rot_vecs2 = rot_vecs_list[mp_inds[1]]
            elif start_fODF_mode[:4] == "both":
                mp1 = mp_list[0][mp_inds[0]][vox]
                mp2 = mp_list[1][mp_inds[1]][vox]
                rot_vecs1 = rot_vecs_list[0][mp_inds[0]]
                rot_vecs2 = rot_vecs_list[1][mp_inds[1]]

            idx1 = np.where(mp1 > 0)
            idx2 = np.where(mp2 > 0)
            
            p_arr = _calc_precision(mp1, mp2, rot_vecs1, rot_vecs2, idx1, idx2,
                                    mp_count, vox, p_arr, precision_type)
                
        mp_count = mp_count + 1

    return p_arr

def predict_grid(data, bvals, bvecs, mask, ad, rd, n, over_sample=None, solver=None,
                                 mean="mean_model", fit_method=None, bounds="preset"):
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
    full_mod = sfm.SparseDeconvolutionModelMultiB(data, bvecs, bvals, mask = mask,
                                                axial_diffusivity = ad,
                                                radial_diffusivity = rd,
                                                over_sample = over_sample,
                                                bounds = bounds, solver = solver,
                                                fit_method = fit_method,
                                                mean = mean, params_file = "temp")
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
             this_data, these_inc0) = ozu.create_combos(bvecs, bvals, data, these_b_inds,
                                                        these_b_inds_rm0, all_inc_0,
                                                        vec_pool, num_choose, combo_num)
            
            mod = sfm.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                        mask = mask, params_file = "temp",
                                                        axial_diffusivity = ad,
                                                        radial_diffusivity = rd,
                                                        bounds = bounds, mean = mean,
                                                        over_sample = over_sample,
                                                        solver = solver)
            sig_out, new_params, inds_arr = new_mean_combos(vec_pool_inds, data, bvals, bvecs,
                                        mask, b_inds, bounds=bounds, these_b_inds=these_b_inds)

            mod.fit_flat_rel_sig_avg = [sig_out, new_params]
            if mean != "empirical":
                mod.regressors = _preload_regressors(si, full_mod, mod, over_sample, mean)
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
        'single': if fitting a single fODF
        'multi': if fitting to multiple fODFs
    
    Returns
    -------
    actual: 2 dimensional array
        Actual signals for the predicted vertices
    predicted: 2 dimensional array 
        Predicted signals for the vertices left out of the fit
    """
    t1 = time.time()
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    _, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = ozu.separate_bvals(bvals,
                                                                    mode = 'remove0')

    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    all_b_idx_rm0 = np.arange(len(all_b_idx))
    
    # Take out the voxels with 0 signal values out of the mask
    S0 = np.mean(data[...,b_inds[0]],-1)
    pre_mask = np.array(mask, dtype=bool)
    ravel_mask = np.ravel(pre_mask)
    
    # Eliminate the voxels where self.S0 == 0.
    ravel_mask[np.where(ravel_mask)[0][np.where(S0[pre_mask] == 0)]] = False
    new_mask = np.reshape(ravel_mask, pre_mask.shape)
    
    actual = np.empty((np.sum(new_mask), len(all_b_idx)))
    predicted = np.empty(actual.shape)
    
    # Generate the regressors in the full model from which we choose the regressors in
    # the reduced model from.
    if fODF_mode == "single": 
        indices = np.array([0])
    elif fODF_mode == "multi":
        indices = np.arange(len(unique_b[1:]))
    
    for bi in indices:
        if fODF_mode == "single":
            all_inc_0 = np.arange(len(rounded_bvals))
            these_b_inds = all_b_idx
            these_b_inds_rm0 = all_b_idx_rm0
        elif fODF_mode == "multi":
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
            this_data, these_inc0) = ozu.create_combos(bvecs, bvals_pool, data, these_b_inds,
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
                  solver = "nnls", mode = None, mean = "mean_model"):
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
        predicted12, predicted22, predicted21] = kfold_xval(data, bvals, bvecs, mask, ad, rd, n,
                                                           fODF_mode = "multi", b_idx1 = b_idx1,
                                                           b_idx2 = b_idx2, mean = mean, 
                                                           solver = solver)
    else:
        # If you want to predict normally, not using k-fold cross-validation
        mod = sfm.SparseDeconvolutionModelMultiB(data[:,:,:,all_inc_0],
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
    
def fODF_EMD(fODF1, fODF2, bvecs1, bvecs2):
    """
    Calcluates the earth mover's distance between two different fODFs
    """
    pre_sig1 = fODF1[..., None]/np.sum(fODF1)
    pre_sig2 = fODF2[..., None]/np.sum(fODF2)
    
    #Flip bvecs:
    bvecs_list = [bvecs1, bvecs2]
    flipped_bvecs_list = []
    for i in np.arange(2):
        these_bvecs = np.squeeze(bvecs_list[i])
        flipped_bvecs_arr = None
        # If the angle between the bvec and (0,0,1) are greater than 90 degrees, flip
        degs = np.rad2deg(np.arccos(np.dot(np.array((0,0,1)), these_bvecs)))
        if type(degs) is not np.array:
            degs = np.array(degs)[..., None]

        for deg_idx, deg in enumerate(degs):
            if deg > 90:
                if len(degs) == 1:
                    bvec = -1*these_bvecs
                else:
                    bvec = -1*these_bvecs[:, deg_idx]
            else:
                if len(degs) == 1:
                    bvec = these_bvecs
                else:
                    bvec = these_bvecs[:, deg_idx]
            
            if flipped_bvecs_arr != None:
                flipped_bvecs_arr = np.concatenate((flipped_bvecs_arr, bvec[None, ...].T), -1)
            else:
                flipped_bvecs_arr = bvec[None, ...].T
        flipped_bvecs_list.append(flipped_bvecs_arr)

    if np.shape(pre_sig1) == (1,1):
        pre_sig1 = np.reshape(pre_sig1, (1,))

    if np.shape(pre_sig2) == (1,1):
        pre_sig2 = np.reshape(pre_sig2, (1,))

    pre_sig1 = np.concatenate((pre_sig1, np.squeeze(flipped_bvecs_list[0].T)), -1)
    pre_sig2 = np.concatenate((pre_sig2, np.squeeze(flipped_bvecs_list[1].T)), -1)
    
    # Put in openCV array format
    if len(np.shape(pre_sig1)) == 1:
        pre_sig1 = pre_sig1[None]
    if len(np.shape(pre_sig2)) == 1:
        pre_sig2 = pre_sig2[None]

    sig1 = cv.fromarray(np.require(np.float32(pre_sig1), requirements='CA'))
    sig2 = cv.fromarray(np.require(np.float32(pre_sig2), requirements='CA'))
    
    EMD = cv.CalcEMD2(sig1, sig2, cv.CV_DIST_L1)
    return EMD
