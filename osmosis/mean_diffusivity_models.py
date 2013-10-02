import scipy.optimize as opt
import inspect
import osmosis.utils as ozu
from osmosis.predict_n import create_combos
import numpy as np

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
            this_sig[b_idx] = np.int(bval)*D
        elif bval*D < c:
            this_sig[b_idx] = c
            
    return this_sig
        
def two_decaying_exp(b, a, D1, D2):
    """
    Decaying exponential and a decaying exponential plus a constant.
    """
    
    this_sig = np.zeros(b.shape)
    
    for b_idx, bval in enumerate(b):
        this_sig[b_idx] = np.max([b*D1, a + b*D2])
        
    return this_sig
    
def optimize_MD_params(data, bvals, mask, func, initial):
    """
    Finds the parameters of the given function to the given data
    that minimizes the sum squared errors.
    """

    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    b = bvals[all_b_idx]/1000
    flat_data = data[np.where(mask)]
    
    param_num = len(inspect.getargspec(func)[0])
    param_out = np.zeros((int(np.sum(mask)), param_num - 1))
    ss_err = ozu.nans(np.sum(mask))
    
    for vox in np.arange(np.sum(mask)).astype(int):
        s0 = np.mean(flat_data[vox, b_inds[0]], -1)
        s_prime = np.log(flat_data[vox, all_b_idx]/s0)
        params, _ = opt.leastsq(err_func, initial, args=(b, s_prime, func))
        
        param_out[vox] = np.squeeze(params)
        ss_err[vox] = np.sum((s_prime - func(b, *params))**2)
        
    return param_out, ss_err
    
def kfold_xval_MD_mod(data, bvals, bvecs, mask, func, initial, n):
    """
    Finds the parameters of the given function to the given data
    that minimizes the sum squared errors using kfold cross validation.
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    b_scaled = bvals/1000
    
    ss_err = np.zeros(int(np.sum(mask)))
    num_choose = (n/100.)*len(all_b_idx)
    vec_pool = np.arange(len(all_b_idx))
    np.random.shuffle(vec_pool)
    all_inc_0 = np.arange(len(rounded_bvals))
    
    flat_data = data[np.where(mask)]
    predict_out = np.zeros((int(np.sum(mask)),len(all_b_idx)))
    
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
            s0 = np.mean(flat_data[vox, b_inds[0]], -1)
            these_b = b_scaled[vec_combo]
            
            s_prime_predict = np.log(flat_data[vox, vec_combo]/s0)
            s_prime_fit = np.log(flat_data[vox, these_inc0]/s0)
            
            params, _ = opt.leastsq(err_func, initial, args=(b_scaled[these_inc0],
                                                               s_prime_fit, func))
            
            predict = func(these_b, *params)
            predict_out[vox, vec_combo_rm0] = func(these_b, *params)
            ss_err[vox] = np.sum((s_prime_predict - predict)**2)

    return ss_err, predict_out