import scipy.optimize as opt
import inspect
import osmosis.utils as ozu
import osmosis.leastsqbound as lsq
import numpy as np

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
    
    return si, vec_combo, vec_combo_rm0, these_bvecs, these_bvals, this_data, these_inc0
    
def err_func(params, b, s_prime, func):
    """
    Error function for fitting a function
    
    Parameters
    ----------
    params : tuple
        A tuple with the parameters of `func` according to their order of input

    b : float array 
        An independent variable. 
    
    s_prime : float array
        The dependent variable. 
    
    func : function
        A function with inputs: `(x, *params)`
    
    Returns
    -------
    The marginals of the fit to b/s_prime given the params
    """
    return s_prime - func(b, *params)
        
def decaying_exp(b, D):
    """
    Just one decaying exponential representation of the mean diffusivity.
    """
    return b*D
        
def decaying_exp_plus_const(b, c, D):
    """
    One decaying exponential representation of the mean diffusivity
    """
    this_sig = np.zeros(b.shape)
    
    for b_idx, bval in enumerate(b):
        if bval*D > c:
            this_sig[b_idx] = bval*D
        elif bval*D < c:
            this_sig[b_idx] = c
            
    return this_sig
        
def two_decaying_exp(b, a, D1, D2):
    """
    Decaying exponential and a decaying exponential plus a constant.
    """
    
    this_sig = np.zeros(b.shape)
    
    for b_idx, bval in enumerate(b):
        this_sig[b_idx] = np.max([bval*D1, a + bval*D2])
        
    return this_sig
    
def two_decaying_exp_plus_const(b, a, c, D1, D2):
    """
    Decaying exponential and a decaying exponential plus a constant.
    """
    
    this_sig = np.zeros(b.shape)
    
    for b_idx, bval in enumerate(b):
        this_sig[b_idx] = np.max([bval*D1, a + bval*D2, c])
        
    return this_sig
    
def optimize_MD_params(data, bvals, mask, func, initial, bounds = None):
    """
    Finds the parameters of the given function to the given data
    that minimizes the sum squared errors.
    """

    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    
    if 0 in bvals:
        all_b_idx = np.squeeze(np.where(bvals != 0))
        b0_inds = np.where(bvals == 0)
    else:
        all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
        b0_inds = b_inds[0]
        
    b = bvals[all_b_idx]/1000
    flat_data = data[np.where(mask)]
    
    param_num = len(inspect.getargspec(func)[0])
    
    # Pre-allocate the outputs:
    param_out = np.zeros((int(np.sum(mask)), param_num - 1))
    ss_err = ozu.nans(np.sum(mask))
    fit_out = ozu.nans(ss_err.shape + (len(all_b_idx),))
    
    for vox in np.arange(np.sum(mask)).astype(int):
        s0 = np.mean(flat_data[vox, b0_inds], -1)
        s_prime = np.log(flat_data[vox, all_b_idx]/s0)
        
        if bounds == None:
            params, _ = opt.leastsq(err_func, initial, args=(b, s_prime, func))
        else:
            lsq_b_out = lsq.leastsqbound(err_func, initial,
                                         args=(b, s_prime, func),
                                         bounds = bounds)
            params = lsq_b_out[0]
        
        param_out[vox] = np.squeeze(params)
        fit_out[vox] = func(b, *params)
        ss_err[vox] = np.sum((s_prime - fit_out[vox])**2)
        
    return param_out, fit_out, ss_err
    
def kfold_xval_MD_mod(data, bvals, bvecs, mask, func, initial, n, bounds = None):
    """
    Finds the parameters of the given function to the given data
    that minimizes the sum squared errors using kfold cross validation.
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    
    if 0 in bvals:
        all_b_idx = np.squeeze(np.where(bvals != 0))
        b0_inds = np.where(bvals == 0)
    else:
        all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
        b0_inds = b_inds[0]
        
    b_scaled = bvals/1000
    flat_data = data[np.where(mask)]
    
    # Pre-allocate outputs
    ss_err = np.zeros(int(np.sum(mask)))
    predict_out = np.zeros((int(np.sum(mask)),len(all_b_idx)))
    
    # Setting up for creating combinations of directions for kfold cross
    # validation:
    
    # Number of directions to leave out at a time.
    num_choose = (n/100.)*len(all_b_idx)
    
    # Find the indices to all the non-b = 0 directions and shuffle them.
    vec_pool = np.arange(len(all_b_idx))
    np.random.shuffle(vec_pool)
    all_inc_0 = np.arange(len(rounded_bvals))
    
    for combo_num in np.arange(np.floor(100./n)):
        (si, vec_combo, vec_combo_rm0,
         these_bvecs, these_bvals,
         this_data, these_inc0) = create_combos(bvecs, bvals,
                                                data, all_b_idx,
                                                np.arange(len(all_b_idx)),
                                                all_b_idx, vec_pool,
                                                num_choose, combo_num)
        this_flat_data = this_data[np.where(mask)]
        
        for vox in np.arange(np.sum(mask)).astype(int):
            s0 = np.mean(flat_data[vox, b0_inds], -1)
            these_b = b_scaled[vec_combo]
            
            s_prime_fit = np.log(flat_data[vox, these_inc0]/s0)
            
            params, _ = opt.leastsq(err_func, initial, args=(b_scaled[these_inc0],
                                                               s_prime_fit, func))
            if bounds == None:
                params, _ = opt.leastsq(err_func, initial, args=(b_scaled[these_inc0],
                                                               s_prime_fit, func))
            else:
                lsq_b_out = lsq.leastsqbound(err_func, initial,
                                             args = (b_scaled[these_inc0],
                                                     s_prime_fit, func),
                                             bounds = bounds)
            params = lsq_b_out[0]
            
            predict = func(these_b, *params)
            predict_out[vox, vec_combo_rm0] = func(these_b, *params)
    
    s0 = np.mean(flat_data[:, b0_inds], -1).astype(float)
    s_prime_predict = np.log(flat_data[:, all_b_idx]/s0[..., None])
    ss_err = np.sum((s_prime_predict - predict_out)**2, -1)

    return ss_err, predict_out