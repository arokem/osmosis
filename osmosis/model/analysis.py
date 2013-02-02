"""

Functions for analyzing one model or (usually) more.

"""
import numpy as np

import osmosis.utils as ozu

def overfitting_index(model1, model2):
    """
    How badly is the model overfitting? This can be assessed by comparing the
    RMSE of the model compared to the fit data (or learning set), relative to
    the RMSE of the model on another data set (or testing set)
    """
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    fit1 = model1.fit[model1.mask]
    fit2 = model2.fit[model2.mask]

    rmse_train1 = ozu.rmse(fit1, sig1)
    rmse_train2 = ozu.rmse(fit2, sig2)

    rmse_test1 = ozu.rmse(fit1, sig2)
    rmse_test2 = ozu.rmse(fit2, sig1)

    fit_rmse = (rmse_train1 + rmse_train2) / 2.
    predict_rmse = (rmse_test1 + rmse_test2) /2. 

    # The measure is a contrast index of the error on the training data vs. the
    # error on the testing data:
    overfit = (fit_rmse - predict_rmse) / (fit_rmse + predict_rmse) 
    
    out = ozu.nans(model1.shape[:-1])    
    out[model1.mask] = overfit

    return out
    
def relative_mae(model1, model2):
    """
    Given two model objects, compare the model fits to signal-to-signal
    reliability in the mean absolute error sense
    """
    # Assume that the last dimension is the signal dimension, so the dimension
    # across which the mae will be calculated: 
    out = ozu.nans(model1.shape[:-1])
    
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    fit1 = model1.fit[model1.mask]
    fit2 = model2.fit[model2.mask]

    signal_mae = ozu.mae(sig1, sig2)
    fit1_mae = ozu.mae(fit1, sig2)
    fit2_mae = ozu.mae(fit2, sig1)

    # Average in each element:
    fit_mae = (fit1_mae + fit2_mae) / 2.

    rel_mae = fit_mae/signal_mae

    out[model1.mask] = rel_mae

    return out


def rsquared(model1, model2):
    """
    Compare two models by way of the Pearson correlation coefficient. For each
    voxel in the mask, average the r squared from model1 prediction to model2
    signal and vice versa.
    """
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    fit1 = model1.fit[model1.mask]
    fit2 = model2.fit[model2.mask]

    out_flat = np.empty(fit1.shape[0])
    
    for vox in xrange(fit1.shape[0]):
        out_flat[vox] = np.mean([np.corrcoef(fit1[vox], sig2[vox])[0,1],
                                 np.corrcoef(fit2[vox], sig1[vox])[0,1]])
        
    out = ozu.nans(model1.shape[:-1])

    out[model1.mask] = out_flat
    return out
    
def cross_predict(model1, model2):
    """
    Given two model objects, fit to the measurements conducted in one, and then
    predict the measurements in the other model's rotational coordinate frame
    (b vectors). Calculate relative RMSE on that prediction, relative to the
    noise in the measurement due to the rotation. Average across both
    directions of this operation.

    """

    out = ozu.nans(model1.shape[:-1])
    
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    # Cross predict, using the parameters from one model to predict the
    # measurements in the other model's b vectors:
    predict1 = model1.predict(model2.bvecs[:, model2.b_idx])[model1.mask]
    predict2 = model2.predict(model1.bvecs[:, model1.b_idx])[model2.mask]

    signal_rmse = ozu.rmse(sig1, sig2)
    predict1_rmse = ozu.rmse(predict1, sig2)
    predict2_rmse = ozu.rmse(predict2, sig1)

    # Average in each element:
    predict_rmse = (predict1_rmse + predict2_rmse) / 2.
    rel_rmse = predict_rmse/signal_rmse

    out[model1.mask] = rel_rmse

    return out




def relative_rmse(model1, model2):
    """
    Given two model objects, compare the model fits to signal-to-signal
    reliability in the root mean square error sense.

    Parameters
    ----------
    model1, model2: two objects from a class inherited from BaseModel

    Returns
    -------
    relative_RMSE: A measure of goodness of fit, relative to measurement
    reliability. The measure is larger than 1 when the model is worse than
    signal-to-signal reliability and smaller than 1 when the model is better. 

    Notes
    -----
    More specificially, we calculate the rmse from the fit in model1 to the
    signal in model2 and then vice-versa. We average between the two
    results. Then, we calculate the rmse from the signal in model1 to the
    signal in model2. We normalize the average model-to-signal rmse to the
    signal-to-signal rmse as a measure of goodness of fit of the model. 

    """
    # Assume that the last dimension is the signal dimension, so the dimension
    # across which the rmse will be calculated: 
    out = ozu.nans(model1.shape[:-1])
    
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    fit1 = model1.fit[model1.mask]
    fit2 = model2.fit[model2.mask]

    signal_rmse = ozu.rmse(sig1, sig2)
    fit1_rmse = ozu.rmse(fit1, sig2)
    fit2_rmse = ozu.rmse(fit2, sig1)

    # Average in each element:
    fit_rmse = (fit1_rmse + fit2_rmse) / 2.

    rel_rmse = fit_rmse/signal_rmse

    out[model1.mask] = rel_rmse

    return out


def noise_ceiling(model1, model2, n_sims=1000, alpha=0.05):
    """
    Calculate the maximal model accuracy possible, given the noise in the
    signal. This is based on the method described by Kay et al. (in review).


    Parameters
    ----------
    model1, model2: two objects from a class inherited from BaseModel

    n_sims: int
       How many simulations of the signal to perform in each voxel.

    alpha:

    Returns
    -------
    coeff: The medians of the distributions of simulated signals in each voxel
    lb, ub: the (1-alpha) confidence interval boundaries on coeff

    Notes
    -----

    The following is performed on the relative signal ($\frac{S}{S_0}$):

    The idea is that noise in the signal can be computed in each voxel by
    comparing the signal in each direction as measured in two different
    measurements. The standard error of measurement for two sample points is
    calculated as follows. First we calculate the mean: 

    .. math ::

       \bar{x_i} = x_{i,1} + x_{i,2} 

    Where $i$ denotes the direction within the voxel. Next, we can calculate
    the standard deviation of the noise: 

    .. math::

        \sigma^2_{noise,i} = \frac{(x_{i,1} - \bar{x_i})^2 + (x_{i,2} - \bar{x_i})^2}{2}

    Note that this is also the standard error of the measurement, since it
    implicity contains the factor of $\sqrt{N-1} = \sqrt{1}$.

    We calculate a mean across directions:

    .. math::

        \sigma_{noise} = \sqrt{mean(\sigma^2_{noise, i})}

    Next, we calculate an estimate of the standard deviation attributable to
    the signal. This is done by first subtracting the noise variance from the
    overall data variance, while making sure that this quantity is non-negative
    and then taking the square root of the resulting quantity:
    
    .. math::

        \sigma_{signal} = \sqrt{max(0, np.mean(\sigma{x_1}, sigma{x_2}) - \sigma^2_{noise})}

    Then, we use Monte Carlo simulation to create a signal: to do that, we
    assume that the signal itself is generated from a Gaussian distribution
    with the above calculated variance). We add noise to this signal (zero mean
    Gaussian with variance $\sigma_{noise}) and compute the correlation between
    the noise-corrupted simulated signal and the noise-free simulated
    signal. The median of the resulting value over many simulations is the
    noise ceiling. The 95% central values represent a confidence interval on
    this value.  

    This is performed over each voxel in the mask.

    """
    # Extract the relative signal 
    sig1 = model1.relative_signal[model1.mask]
    sig2 = model2.relative_signal[model1.mask]

    noise_ceil_flat = np.empty(sig1.shape[0])
    ub_flat = np.empty(sig1.shape[0])
    lb_flat = np.empty(sig1.shape[0])

    for vox in xrange(sig1.shape[0]):
        sigma_noise = np.sqrt(np.mean(np.var([sig1[vox],sig2[vox]],0)))
        mean_sig_w_noise = np.mean([sig1[vox], sig2[vox]], 0)
        var_sig_w_noise = np.var(mean_sig_w_noise)
        sigma_signal = np.sqrt(np.max([0, var_sig_w_noise - sigma_noise**2]))
        # Create the simulated signal over many iterations:
        sim_signal = sigma_signal * np.random.randn(sig1[vox].shape[0] * n_sims)
        sim_signal_w_noise = (sim_signal +
                    sigma_noise * np.random.randn(sig1[vox].shape[0] * n_sims))

        # Reshape it so that you have n_sims separate simulations of this voxel:
        sim_signal = np.reshape(sim_signal, (n_sims, -1))
        sim_signal_w_noise = np.reshape(sim_signal_w_noise, (n_sims, -1))
        coeffs = ozu.coeff_of_determination(sim_signal_w_noise, sim_signal)
        sort_coeffs = np.sort(coeffs)
        lb_flat[vox] = sort_coeffs[alpha/2 * coeffs.shape[-1]]
        ub_flat[vox] = sort_coeffs[1-alpha/2 * coeffs.shape[-1]]
        noise_ceil_flat[vox] = np.median(coeffs)

    out_coeffs = ozu.nans(model1.mask.shape)
    out_ub = ozu.nans(out_coeffs.shape)
    out_lb = ozu.nans(out_coeffs.shape)
    
    out_coeffs[model1.mask] = noise_ceil_flat
    out_lb[model1.mask] = lb_flat
    out_ub[model1.mask] = ub_flat

    return out_coeffs, out_lb, out_ub


        
def coeff_of_determination(model1, model2):
    """
    Calculate the voxel-wise coefficient of determination between on model fit
    and the other model signal, averaged across both ways.
    """
    out = ozu.nans(model1.shape[:-1])
    
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    fit1 = model1.fit[model1.mask]
    fit2 = model2.fit[model2.mask]

    fit1_R_sq = ozu.coeff_of_determination(fit1, sig2, axis=-1)
    fit2_R_sq = ozu.coeff_of_determination(fit2, sig1, axis=-1)

    # Average in each element:
    fit_R_sq = np.mean([fit1_R_sq, fit2_R_sq],0)

    out[model1.mask] = fit_R_sq

    return out


def rmse(model1, model2):
    """
    Calculate the voxel-wise RMSE between one model signal and the other model
    signal. 
    """
    out = ozu.nans(model1.shape[:-1])
    
    sig1 = model1.signal[model1.mask]
    sig2 = model2.signal[model2.mask]
    out_flat = np.empty(sig1.shape[0])
    
    for vox in sig1.shape[0]:
        out_flat[vox] = ozu.rmse(sig1[vox], sig2[vox])

    out[model1.mask] = out_flat
    return out


def pdd_reliability(model1, model2):
    """

    Compute the angle between the first PDD in two models.

    Parameters
    ----------
    model1, model2: two objects from a class inherited from BaseModel.
       Must implement an auto_attr class method 'principal_diffusion_direction',
       which returns arrays of 3-vectors representing a direction in three space
       which is the principal diffusion direction in each voxel. Some models
       will have more than one principal diffusion direction in each voxel. In
       that case, the first direction in each voxel will be used to represent
       that voxel. 
    
    """
    vol_shape = model1.shape[:3]
    pdd1 = model1.principal_diffusion_direction[model1.mask]
    pdd2 = model2.principal_diffusion_direction[model2.mask]

    # Some models create not only the first PDD, but subsequent directions as
    # well, so If we have a lot of PDD, we take only the first one: 
    if len(pdd1.shape) == 3:
        pdd1 = pdd1[:, 0]
    if len(pdd2.shape) == 3:
        pdd2 = pdd2[:, 0]

    out_flat = np.empty(pdd1.shape[0])    
    for vox in xrange(pdd1.shape[0]):
        this_ang = np.rad2deg(ozu.vector_angle(pdd1[vox], pdd2[vox]))
        out_flat[vox] = np.min([this_ang, 180-this_ang])

    out = ozu.nans(vol_shape)
    out[model1.mask] = out_flat
    return out


def model_params_reliability(model1, model2):
    """
    Compute the vector angle between the sets of model params for two model
    instances in each voxel as a measure of model reliability.
    """
    vol_shape = model1.shape[:3]

    mp1 = model1.model_params[model1.mask]
    mp2 = model2.model_params[model1.mask]
    
    out_flat = np.empty(mp1.shape[0])
    
    for vox in xrange(out_flat.shape[0]):
        out_flat[vox]= np.rad2deg(ozu.vector_angle(mp1[vox], mp2[vox]))

    out = ozu.nans(vol_shape)
    out[model1.mask] = out_flat
    return out

def fit_reliability(model1, model2):
    """
    Compute the vector angle between the model-predicted signal in each voxel
    as a measure of model reliability. 
    """
    vol_shape = model1.shape[:3]

    fit1 = model1.fit[model1.mask]
    fit2 = model2.fit[model1.mask]
    
    out_flat = np.empty(fit1.shape[0])
    
    for vox in xrange(out_flat.shape[0]):
        out_flat[vox]= np.corrcoef(fit1[vox], fit2[vox])[0,1]
        

    out = ozu.nans(vol_shape)
    out[model1.mask] = out_flat
    return out
