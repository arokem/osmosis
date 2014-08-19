"""
Various functions for calculating various attributes of different isotropic
models.
"""

import scipy.optimize as opt
import inspect
import osmosis.utils as ozu
import osmosis.model.dti as dti
import osmosis.leastsqbound as lsq
import numpy as np

# rs within function names means relative signal
# nf within function names means noise floor

def err_func(params, b, s_prime, func):
    """
    Error function for fitting a function

    Parameters
    ----------
    params: tuple
        A tuple with the parameters of `func` according to their order of input
    b: float array
        An independent variable (b-values)
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
    Just one decaying exponential representation of the mean diffusivity.  Used
    if fitting to the log of the relative signal.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    D: float array
        The fitted diffusivities.

    Returns
    -------
    The fitted diffusion signal in log form
    """
    return b*D


def decaying_exp_plus_const(b, c, D):
    """
    One decaying exponential plus a noise floor constant.  Used if fitting to
    the log of the relative signal.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    c: float array
        Noise floor constant at each voxel.
    D: float array
        The fitted diffusivities.

    Returns
    -------
    The fitted diffusion signal in log form
    """

    b_extra_dim = b[..., None]
    this_sig = np.max(np.concatenate([b_extra_dim*D, c*np.ones((len(b),1))],
                                                                     -1),-1)

    return this_sig


def two_decaying_exp(b, a, D1, D2):
    """
    Two decaying exponentials.  Used if fitting to the log of the
    relative signal.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    a: float array
        The fraction of the signal from one diffusivity versus another.
    D1: float array
        The first fitted diffusivities.
    D2: float array
        The second fitted diffusivities.

    Returns
    -------
    The fitted diffusion signal in log form
    """

    b_extra_dim = b[..., None]
    this_sig = np.max(np.concatenate([b_extra_dim*D1, a + b_extra_dim*D2],
                                                                   -1),-1)

    return this_sig


def two_decaying_exp_plus_const(b, a, c, D1, D2):
    """
    Two decaying exponentials plus a constant.  Used if fitting to the log of
    the relative signal.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    c: float array
        Noise floor constant at each voxel.
    D1: float array
        The first fitted diffusivities.
    D2: float array
        The second fitted diffusivities.

    Returns
    -------
    The fitted diffusion signal in log form
    """

    b_extra_dim = b[..., None]
    this_sig = np.max(np.concatenate([b_extra_dim * D1, a + b_extra_dim * D2,
                                          c*np.ones((len(b),1))],-1),-1)

    return this_sig


def single_exp_rs(b, D):
    """
    Relative signal of the decaying exponential.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    D: float array
        The fitted diffusivities.

    Returns
    -------
    The fitted relative diffusion signal
    """
    return np.exp(-b * D)


def single_exp_nf_rs(b, nf, D):
    """
    Relative signal of the decaying exponential plus a noise floor.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    nf: float array
        Noise floor constant at each voxel.
    D: float array
        The fitted diffusivities.

    Returns
    -------
    The fitted relative diffusion signal
    """
    return np.exp(-b*D) + nf

def bi_exp_rs(b, f1, D1, D2):
    """
    Constrained bi-exponential model plus noise floor constant.

    Parameters
    ----------
    b: float array
        An independent variable (b-values)
    D1: float array
        The first fitted diffusivities.
    D2: float array
        The second fitted diffusivities.
    f1: float array
        The fraction of the signal from D1.  The fraction of the signal from
        D2 is 1 - f1.

    Returns
    -------
    The fitted relative diffusion signal
    """
    rel_sig = f1*np.exp(-b*D1) + (1-f1)*np.exp(-b*D2)

    return rel_sig

def bi_exp_nf_rs(b, nf, f1, D1, D2):
    """
    Constrained bi-exponential model plus noise floor constant.

    Parameters
    ----------
    D1: float array
        The first fitted diffusivities.
    D2: float array
        The second fitted diffusivities.
    f1: float array
        The fraction of the signal from D1.  The fraction of the signal from
        D2 is 1 - f1.

    Returns
    -------
    The fitted relative diffusion signal
    """
    rel_sig = f1*np.exp(-b*D1) + (1-f1)*np.exp(-b*D2) + nf

    return rel_sig

def initial_params(data, bvecs, bvals, model, mask=None, params_file='temp'):
    """
    Determine the initial values for fitting the isotropic diffusion model.
    This only works on the models that fit to the relative diffusion signal.

    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvecs: 2 dimensional array
        All the b vectors
    bvals: 1 dimensional array
        All b values
    model: str
        Isotropic model
    mask: 3 dimensional array
        Mask of the data
    params_file: obj or str
        File handle of the param_files containing the tensor parameters.

    Returns
    -------
    bounds: list
        A list containing the bounds for each parameter for least squares
        fitting.
    initial: list
        A list containing the initial values for each parameter for least
        squares fitting.
    """
    dti_mod = dti.TensorModel(data, bvecs, bvals, mask=mask,
                                    params_file=params_file)

    d = dti_mod.mean_diffusivity[np.where(mask)]

    # Find initial noise floor
    _, b_inds, _, _ = ozu.separate_bvals(bvals)
    b0_data = data[np.where(mask)][:, b_inds[0]]
    nf = np.std(b0_data, -1)/np.mean(b0_data, -1)

    if model == single_exp_rs:
        bounds = [(0, 4)]
        initial = d

    elif model == single_exp_nf_rs:
        bounds = [(0, 10000), (0, 4)]
        initial = np.concatenate([nf[..., None], d[...,None]], -1)

    elif model== bi_exp_rs:
        bounds = [(0, 1), (0, 4), (0, 4)]
        initial = np.concatenate([0.5*np.ones((len(d),1)), d[...,None],
                                                      d[...,None]], -1)
    elif model== bi_exp_nf_rs:
        bounds = [(0, 10000), (0, 1), (0, 4), (0, 4)]
        initial = np.concatenate([nf[..., None], 0.5*np.ones((len(d),1)),
                                           d[...,None], d[...,None]], -1)
    return bounds, initial


def _diffusion_inds(bvals, b_inds, rounded_bvals):
    """
    Extracts the diffusion-weighted and non-diffusion weighted indices.

    Parameters
    ----------
    bvals: 1 dimensional array
        All b values
    b_inds: list
        List of the indices corresponding to the separated b values.  Each index
        contains an array of the indices to the grouped b values with similar
        values
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


def isotropic_params(data, bvals, bvecs, mask, func, factor=1000,
                       initial="preset", bounds="preset", params_file='temp',
                       signal="relative_signal"):
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
    func: str or callable
        String indicating the mean model function to perform kfold
        cross-validation on.
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
    if isinstance(func, str):
        # Grab the function handle for the desired mean model
        func = globals()[func]
    

    # Get the initial values for the desired mean model
    if (bounds == "preset") | (initial == "preset"):
        all_params = initial_params(data, bvecs, bvals, func, mask=mask,
                                    params_file=params_file)
    if bounds == "preset":
        bounds = all_params[0]
    if initial == "preset":
        func_initial = all_params[1]
    else:
        this_initial = initial

    # Separate b values and grab their indices
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_idx, b0_inds = _diffusion_inds(bvals, b_inds, rounded_bvals)

    # Divide the b values by a scaling factor first.
    b = bvals[all_b_idx]/factor
    flat_data = data[np.where(mask)]

    # Get the number of inputs to the mean diffusivity function
    param_num = len(inspect.getargspec(func)[0])

    # Pre-allocate the outputs:
    param_out = np.zeros((int(np.sum(mask)), param_num - 1))
    cod = ozu.nans(np.sum(mask))
    fit_out = ozu.nans(cod.shape + (len(all_b_idx),))

    for vox in np.arange(np.sum(mask)).astype(int):
        s0 = np.mean(flat_data[vox, b0_inds], -1)

        if initial == "preset":
            this_initial = func_initial[vox]

        input_signal = flat_data[vox, all_b_idx]/s0
        if signal == "log":
            input_signal = np.log(input_signal)

        if bounds == None:
            params, _ = opt.leastsq(err_func, this_initial, args=(b,
                                                input_signal, func))
        else:
            lsq_b_out = lsq.leastsqbound(err_func, this_initial,
                                         args=(b, input_signal, func),
                                         bounds = bounds)
            params = lsq_b_out[0]

        param_out[vox] = np.squeeze(params)
        fit_out[vox] = func(b, *params)
        cod[vox] = ozu.coeff_of_determination(input_signal, fit_out[vox])

    return param_out, fit_out, cod

def kfold_xval_MD_mod(data, bvals, bvecs, mask, func, n, factor = 1000,
                      initial="preset", bounds = "preset", params_file='temp',
                      signal="relative_signal"):
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
    cod: 1 dimensional array
        Coefficent of Determination between data and predicted values
    predicted: 2 dimensional array
        Predicted mean for the vertices left out of the fit
    """
    if isinstance(func, str):
        # Grab the function handle for the desired mean model
        func = globals()[func]

    # Get the initial values for the desired mean model
    if (bounds == "preset") | (initial == "preset"):
        all_params = initial_params(data, bvecs, bvals, func, mask=mask,
                                    params_file=params_file)
    if bounds == "preset":
        bounds = all_params[0]
    if initial == "preset":
        func_initial = all_params[1]
    else:
        this_initial = initial

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
         vec_pool_inds, these_bvecs, these_bvals,
         this_data, these_inc0) = ozu.create_combos(bvecs, bvals, data,
                                                    all_b_idx,
                                                    np.arange(len(all_b_idx)),
                                                    all_b_idx, vec_pool,
                                                    num_choose, combo_num)
        this_flat_data = this_data[np.where(mask)]

        for vox in np.arange(np.sum(mask)).astype(int):
            s0 = np.mean(flat_data[vox, b0_inds], -1)
            these_b = b_scaled[vec_combo] # b values to predict

            if initial == "preset":
                this_initial = func_initial[vox]

            input_signal = flat_data[vox, these_inc0]/s0

            if signal == "log":
                input_signal = np.log(input_signal)

            # Fit mean model to part of the data
            params, _ = opt.leastsq(err_func, this_initial,
                                    args=(b_scaled[these_inc0],
                                    input_signal, func))
            if bounds == None:
                params, _ = opt.leastsq(err_func, this_initial,
                                        args=(b_scaled[these_inc0],
                                        input_signal, func))
            else:
                lsq_b_out = lsq.leastsqbound(err_func, this_initial,
                                             args = (b_scaled[these_inc0],
                                                     input_signal, func),
                                             bounds = bounds)
                params = lsq_b_out[0]
            # Predict the mean values of the left out b values using the
            # parameters from fitting to part of the b values.
            predict = func(these_b, *params)
            predict_out[vox, vec_combo_rm0] = func(these_b, *params)

    # Find the relative diffusion signal.
    s0 = np.mean(flat_data[:, b0_inds], -1).astype(float)
    input_signal = flat_data[:, all_b_idx]/s0[..., None]
    if signal == "log":
        input_signal = np.log(input_signal)
    cod = ozu.coeff_of_determination(input_signal, predict_out, axis=-1)

    return cod, predict_out
