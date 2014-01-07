
import os

import numpy as np

import nibabel as ni
import dipy.reconst.dti as dti
import dipy.core.gradients as gradients

from osmosis.model.base import BaseModel, SCALE_FACTOR
import osmosis.descriptors as desc
import osmosis.utils as ozu
import osmosis.tensor as ozt
import osmosis.boot as boot


class TensorModel(BaseModel):

    """
    A class for representing and solving a simple forward model. Just the
    diffusion tensor.
    
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 params_file=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 verbose=True,
                 fit_method='WLS'):
        """
        Parameters
        -----------

        data, bvecs, bvals: see DWI inputs

        scaling_factor: This scales the b value for the Stejskal/Tanner
        equation

        mask: ndarray or file-name
              An array of the same shape as the data, containing a binary mask
              pointing to the locations of voxels that should be analyzed.

        sub_sample: int or array of ints.

           If we want to sub-sample the DWI data on the sphere (in the bvecs),
           we can do one of two things: 

           1. If sub_sample is an integer, that number of random bvecs will be
           chosen from the data.

           2. If an array of indices is provided, these will serve as indices
           into the last dimension of the data and only that part of the data
           will be used


        params_file: A file to cache the initial tensor calculation in. If this
        file already exists, we pull the tensor fit out of it. Otherwise, we
        calculate the tensor fit and save this file with the params of the
        tensor fit.

        fit_method: str
           'WLS' for weighted least squares fitting (default) or 'LS'/'OLS' for
           ordinary least squares.
        
        """
        # Initialize the super-class:
        BaseModel.__init__(self,
                           data,
                           bvecs,
                           bvals,
                           affine=None,
                           mask=mask,
                           scaling_factor=scaling_factor,
                           sub_sample=sub_sample,
                           params_file=params_file,
                           verbose=verbose) 

        # Allow using 'OLS' to denote the ordinary least-squares method
        if fit_method=='OLS':
            fit_method = 'LS'
        
        self.fit_method = fit_method
        self.gtab = gradients.gradient_table(self.bvals, self.bvecs)
        
    @desc.auto_attr
    def model_params(self):
        """
        The diffusion tensor parameters estimated from the data, using dipy.
        If this calculation has already occurred, just load the data from a
        nifti file, which has shape x by y by z by 12, where the last dimension
        is the model params:

        evecs (9) + evals (3)
        
        """
        out = ozu.nans((self.data.shape[:3] +  (12,)))
        
        flat_params = np.empty((self._flat_S0.shape[0], 12))
        
        # The file already exists: 
        if os.path.isfile(self.params_file):
            if self.verbose:
                print("Loading TensorModel params from: %s" %self.params_file)
            out[self.mask] = ni.load(self.params_file).get_data()[self.mask]
        else:
            if self.verbose:
                print("Fitting TensorModel params using dipy")
            tensor_model = dti.TensorModel(self.gtab,
                                  fit_method=self.fit_method)
            for vox, vox_data in enumerate(self.data[self.mask]):
                flat_params[vox] = tensor_model.fit(vox_data).model_params

            out[self.mask] = flat_params
            # Save the params for future use: 
            params_ni = ni.Nifti1Image(out, self.affine)
            # If we asked it to be temporary, no need to save anywhere: 
            if self.params_file != 'temp':
                params_ni.to_filename(self.params_file)
        # And return the params for current use:
        return out

    @desc.auto_attr
    def evecs(self):
        return np.reshape(self.model_params[..., 3:], 
                          self.model_params.shape[:3] + (3,3))

    @desc.auto_attr
    def evals(self):
        return self.model_params[..., :3]

    @desc.auto_attr
    def mean_diffusivity(self):
        #adc/md = (ev1+ev2+ev3)/3
        return self.evals.mean(-1)

        
    @desc.auto_attr
    def adc_residuals(self):
        """
        The model-predicted ADC, conpared with the empirical ADC.
        """
        return self.model_adc - self.adc
    
    @desc.auto_attr
    def fractional_anisotropy(self):
        """
        .. math::

            FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                        \lambda_3)^2+(\lambda_2-lambda_3)^2}{\lambda_1^2+
                        \lambda_2^2+\lambda_3^2} }

        """
        out = ozu.nans(self.data.shape[:3])
        
        lambda_1 = self.evals[..., 0][self.mask]
        lambda_2 = self.evals[..., 1][self.mask]
        lambda_3 = self.evals[..., 2][self.mask]

        out[self.mask] = ozu.fractional_anisotropy(lambda_1, lambda_2, lambda_3)

        return out

    @desc.auto_attr
    def radial_diffusivity(self):
        return np.mean(self.evals[...,1:],-1)

    @desc.auto_attr
    def axial_diffusivity(self):
        return self.evals[...,0]


    @desc.auto_attr
    def linearity(self):
        out = ozu.nans(self.data.shape[:3])
        out[self.mask] = ozu.tensor_linearity(self.evals[..., 0][self.mask],
                                              self.evals[..., 1][self.mask],
                                              self.evals[..., 2][self.mask])
        return out

    @desc.auto_attr
    def planarity(self):
        out = ozu.nans(self.data.shape[:3])
        out[self.mask] = ozu.tensor_planarity(self.evals[..., 0][self.mask],
                                              self.evals[..., 1][self.mask],
                                              self.evals[..., 2][self.mask])
        return out

    @desc.auto_attr
    def sphericity(self):
        out = ozu.nans(self.data.shape[:3])
        out[self.mask] = ozu.tensor_sphericity(self.evals[..., 0][self.mask],
                                               self.evals[..., 1][self.mask],
                                               self.evals[..., 2][self.mask])
        return out

    # Self Diffusion Tensor, taken from dipy.reconst.dti:
    @desc.auto_attr
    def tensors(self):
        out = ozu.nans(self.evecs.shape)
        evals = self.evals[self.mask]
        evecs = self.evecs[self.mask]
        D_flat = np.empty(evecs.shape)
        for ii in xrange(len(D_flat)):
            Q = evecs[ii]
            L = evals[ii]
            D_flat[ii] = np.dot(Q*L, Q.T)

        out[self.mask] = D_flat
        return out

    @desc.auto_attr
    def mode(self):
        out = ozu.nans(self.data.shape[:3])
        out[self.mask] = dti.tensor_mode(self.tensors)[self.mask]
        return out

    @desc.auto_attr
    def model_adc(self):
        out = np.empty(self.signal.shape)
        tensors_flat = self.tensors[self.mask].reshape((-1,3,3))
        adc_flat = np.empty(self.signal[self.mask].shape)

        for ii in xrange(len(adc_flat)):
            adc_flat[ii] = ozt.apparent_diffusion_coef(
                                        self.bvecs[:,self.b_idx],
                                        tensors_flat[ii])

        out[self.mask] = adc_flat
        return out

    def predict_adc(self, sphere):
        """

        The ADC predicted on a sphere (containing points other than the bvecs)
        
        """
        out = ozu.nans(self.signal.shape[:3] + (sphere.shape[-1],))
        tensors_flat = self.tensors[self.mask].reshape((-1,3,3))
        pred_adc_flat = np.empty((np.sum(self.mask), sphere.shape[-1]))

        for ii in xrange(len(pred_adc_flat)):
            pred_adc_flat[ii] = ozt.apparent_diffusion_coef(sphere,
                                                       tensors_flat[ii])

        out[self.mask] = pred_adc_flat

        return out
        

    @desc.auto_attr
    def fiber_volume_fraction(self):
        """
        Fiber volume fraction estimated using equation 2 in:

        Stikov, N, Perry, LM, Mezer, A, Rykhlevskaia, E, Wandell, BA, Pauly,
        JM, Dougherty, RF (2011) Bound pool fractions complement diffusion
        measures to describe white matter micro and macrostructure. Neuroimage
        54: 1112.
        
        """
        flat_fvf = ozu.fiber_volume_fraction(
            self.fractional_anisotropy[self.mask])

        out = np.empty(self.fractional_anisotropy.shape)
        out[self.mask] = flat_fvf

        return out

    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        The principal diffusion direction in X,Y,Z coordinates on the surface
        of the sphere 
        """
        # It's simply the first eigen-vector
        return self.evecs[...,0]


    @desc.auto_attr
    def fit(self):
        if self.verbose:
            print("Predicting signal from TensorModel")
        adc_flat = self.model_adc[self.mask]
        fit_flat = np.empty(adc_flat.shape)
        out = ozu.nans(self.signal.shape)

        for ii in xrange(len(fit_flat)):
            fit_flat[ii] = ozt.stejskal_tanner(self._flat_S0[ii],
                                               self.bvals[self.b_idx],
                                               adc_flat[ii])

        out[self.mask] = fit_flat
        return out

    def predict(self, sphere):
        """
        Predict the values of the signal on a novel sphere (not neccesarily
        measured points) in every voxel

        Parameters
        ----------
        sphere : 3 x n array
        
        """
        if self.verbose:
            print("Predicting signal from TensorModel")

        pred_adc_flat = self.predict_adc(sphere)[self.mask]
        predict_flat = np.empty(pred_adc_flat.shape)

        out = ozu.nans(self.signal.shape[:3] + (sphere.shape[-1], ))
        # We will assume one b-value use that one below for all the bvecs:
        bval = self.bvals[self.b_idx][0]
        for ii in xrange(len(predict_flat)):
            predict_flat[ii] = ozt.stejskal_tanner(self._flat_S0[ii],
                                        bval*np.ones(pred_adc_flat.shape[-1]),
                                        pred_adc_flat[ii])

        out[self.mask] = predict_flat
        return out

    @desc.auto_attr
    def model_diffusion_distance(self):
        """

        The diffusion distance implied by the model parameters
        """
        tensors_flat = self.tensors[self.mask]
        dist_flat = np.empty(self._flat_signal.shape)        
        for vox in xrange(len(dist_flat)):
            dist_flat[vox]=ozt.diffusion_distance(self.bvecs[:, self.b_idx],
                                                  tensors_flat[vox])
        out = ozu.nans(self.signal.shape)
        out[self.mask] = dist_flat

        return out
            

        
def _dyad_stats(tensor_model_list, mask=None, dyad_stat=boot.dyad_coherence,
                average=True):
    """
    Helper function that does most of the work on calcualting dyad statistics
    """
    if mask is None:
        mask = np.ones(tensor_model_list[0].shape[:3])
        
    # flatten the eigenvectors:
    tensor_model_flat=np.array([this.evecs[np.where(mask)] for this in
    tensor_model_list])
    out_flat = np.empty(tensor_model_flat[0].shape[0])

    # Loop over voxels
    for idx in xrange(tensor_model_flat.shape[1]):
        dyad = boot.dyadic_tensor(tensor_model_flat[:,idx,:,:],
                                  average=average)
        out_flat[idx] = dyad_stat(dyad)

    out = ozu.nans(tensor_model_list[0].shape[:3])
    out[np.where(mask)] = out_flat
    return out        
    

def tensor_coherence(tensor_model_list, mask=None):
    """
    Calculate the coherence of the principle diffusion direction over a bunch
    of TensorModel class instances.


    This is $\kappa = 1-\sqrt{\frac{\beta_2 + \beta_3}{\beta_1}}$, where the
    $\beta_i$ are the eigen-values of the dyadic tensor over the list of
    TensorModel class instances provided as input. 
    """
    # This is the default behavior:
    return _dyad_stats(tensor_model_list, mask=mask,
                       dyad_stat=boot.dyad_coherence,
                       average=True)

def tensor_dispersion(tensor_model_list, mask=None):
    """
    Calculate the dispersion (in degrees) of the principle diffusion direction
    over a sample of TensorModel class instances.
    """
    # Calculate the dyad_dispersion instead:
    return _dyad_stats(tensor_model_list, mask=mask,
                       dyad_stat=boot.dyad_dispersion,
                       average=False) # This one needs to know the individual
                                      # dyads, in addition to the average one.
