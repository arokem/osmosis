import scipy.optimize as opt
import inspect
import osmosis.utils as ozu
import osmosis.leastsqbound as lsq
import numpy as np
  
def err_func(params, b, s_prime, func):
    """
    Error function for fitting a function
    
    Parameters
    ----------
    params: tuple
        A tuple with the parameters of `func` according to their order of input
    b: float array 
        An independent variable. 
    s_prime: float array
        The dependent variable.     
    func: function
        A function with inputs: `(b, *params)`
    
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
    
    b_extra_dim = b[..., None]
    this_sig = np.max(np.concatenate([b_extra_dim*D, c*np.ones((len(b),1))],-1),-1)
    
    return this_sig
        
def two_decaying_exp(b, a, D1, D2):
    """
    Decaying exponential and a decaying exponential plus a constant.
    """
    
    b_extra_dim = b[..., None]
    this_sig = np.max(np.concatenate([b_extra_dim*D1, a + b_extra_dim*D2],-1),-1)
    
    return this_sig
    
def two_decaying_exp_plus_const(b, a, c, D1, D2):
    """
    Decaying exponential and a decaying exponential plus a constant.
    """

    b_extra_dim = b[..., None]
    this_sig = np.max(np.concatenate([b_extra_dim*D1, a + b_extra_dim*D2,
                                          c*np.ones((len(b),1))],-1),-1)
        
    return this_sig

def _diffusion_inds(bvals, b_inds, rounded_bvals):
    """
    Extracts the diffusion-weighted and non-diffusion weighted indices.
    
    Parameters
    ----------
    bvals: 1 dimensional array
        All b values
    b_inds: list
        List of the indices corresponding to the separated b values.  Each index
        contains an array of the indices to the grouped b values with similar values
    rounded_bvals: 1 dimensional array
        B values after rounding
        
    Returns
    -------
    all_b_idx: 1 dimensional array
        Indices corresponding to all non-zero b values
    b0_inds: 1 dimensional array
        Indices corresponding to all b = 0 values
    """
    
    if 0 in bvals:
        # If 0 is in the b values, then there is no need to do rounding
        # in order to separate the b values and determine which ones are
        # close enough to 0 to be considered 0.
        all_b_idx = np.squeeze(np.where(bvals != 0))
        b0_inds = np.where(bvals == 0)
    else:
        all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
        b0_inds = b_inds[0]
    
    return all_b_idx, b0_inds
def optimize_MD_params(data, bvals, mask, func, initial, factor = 1000, bounds = None):
    """
    Finds the parameters of the given function to the given data
    that minimizes the sum squared errors.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    mask: 3 dimensional array
        Brain mask of the data
    func: function handle
        Mean model to perform kfold cross-validation on.
    initial: tuple
        Initial values for the parameters.
    factor: int
        Integer indicating the scaling factor for the b values
    bounds: list
        List containing tuples indicating the bounds for each parameter in
        the mean model function.
        
    Returns
    -------
    param_out: 2 dimensional array
        Parameters that minimize the residuals
    fit_out: 2 dimensional array 
        Model fitted means
    ss_err: 2 dimensional array 
        Sum squared error between the model fitted means and the actual means
    """

    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_idx, b0_inds = _diffusion_inds(bvals, b_inds, rounded_bvals)
    
    # Divide the b values by a scaling factor first.
    b = bvals[all_b_idx]/factor
    flat_data = data[np.where(mask)]
    
    # Get the number of inputs to the mean diffusivity function
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
    
def kfold_xval_MD_mod(data, bvals, bvecs, mask, func, initial, n, factor = 1000, bounds = None):
    """
    Finds the parameters of the given function to the given data
    that minimizes the sum squared errors using kfold cross validation.
    
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
    func: function handle
        Mean model to perform kfold cross-validation on.
    initial: tuple
        Initial values for the parameters.
    n: int
        Integer indicating the percent of vertices that you want to predict
    factor: int
        Integer indicating the scaling factor for the b values
    bounds: list
        List containing tuples indicating the bounds for each parameter in
        the mean model function.
        
    Returns
    -------
    actual: 2 dimensional array
        Actual mean for the predicted vertices
    predicted: 2 dimensional array 
        Predicted mean for the vertices left out of the fit
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_idx, b0_inds = _diffusion_inds(bvals, b_inds, rounded_bvals)
        
    b_scaled = bvals/factor
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
    
    # Start cross-validation
    for combo_num in np.arange(np.floor(100./n)):
        (si, vec_combo, vec_combo_rm0,
         these_bvecs, these_bvals,
         this_data, these_inc0) = ozu.create_combos(bvecs, bvals, data, all_b_idx,
                                                np.arange(len(all_b_idx)),
                                                all_b_idx, vec_pool,
                                                num_choose, combo_num)
        this_flat_data = this_data[np.where(mask)]
        
        for vox in np.arange(np.sum(mask)).astype(int):
            s0 = np.mean(flat_data[vox, b0_inds], -1)
            these_b = b_scaled[vec_combo] # b values to predict
            
            s_prime_fit = np.log(flat_data[vox, these_inc0]/s0)
            # Fit mean model to part of the data
            params, _ = opt.leastsq(err_func, initial, args=(b_scaled[these_inc0],
                                                               s_prime_fit, func))
            if bounds == None:
                params, _ = opt.leastsq(err_func, initial, args=(b_scaled[these_inc0],
                                                               s_prime_fit, func))
            else:
                lsq_b_out = lsq.leastsqbound(err_func, initial,
                                             args = (b_scaled[these_inc0],
                                                     s_prime_fit, func), bounds = bounds)
                params = lsq_b_out[0]
            # Predict the mean values of the left out b values using the parameters from
            # fitting to part of the b values.
            predict = func(these_b, *params)
            predict_out[vox, vec_combo_rm0] = func(these_b, *params)

    s0 = np.mean(flat_data[:, b0_inds], -1).astype(float)
    s_prime_predict = np.log(flat_data[:, all_b_idx]/s0[..., None])
    ss_err = np.sum((s_prime_predict - predict_out)**2, -1)

    return ss_err, predict_out