"""

Sparse spherical deconvolution of diffusion data with multiple b values.

"""


import os
import inspect
import warnings

import numpy as np
# Get stuff from sklearn, if that's available:
try:
    from sklearn.linear_model import Lasso, LassoCV
    # Get other stuff from sklearn.linear_model:
    from sklearn.linear_model import ElasticNet, Lars, Ridge, ElasticNetCV
    # Get OMP:
    from sklearn.linear_model.omp import OrthogonalMatchingPursuit as OMP

    has_sklearn = True

    # Make a dict with solvers to be used for choosing among them:
    sklearn_solvers = dict(Lasso=Lasso,
                           OMP=OMP,
                           ElasticNet=ElasticNet,
                           ElasticNetCV=ElasticNetCV,
                           Lars=Lars)

except ImportError:
    e_s = "Could not import sklearn. Download and install from XXX"
    warnings.warn(e_s)
    has_sklearn = False    


import nibabel as ni
import dipy.reconst.recspeed as recspeed
import dipy.core.sphere as dps
import dipy.core.geometry as geo

import osmosis.utils as ozu
import osmosis.descriptors as desc
import osmosis.cluster as ozc
import osmosis.tensor as ozt
from osmosis.snr import separate_bvals

from osmosis.model.sparse_deconvolution import SparseDeconvolutionModel
#from osmosis.model.canonical_tensor import AD, RD
from osmosis.model.base import SCALE_FACTOR
from osmosis.model.io import params_file_resolver


# For now, let's assume 3 bvalues and let's assume these are the diffusivities:
AD = [1.3, 1.5, 1.8]
RD = [0.8, 0.5, 0.3]

class SparseDeconvolutionModelMultiB(SparseDeconvolutionModel):
    """
    Use Elastic Net to do spherical deconvolution with a canonical tensor basis
    set. 
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 solver=None,
                 solver_params=None,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 mode='relative_signal',
                 verbose=True):
        """
        Initialize SparseDeconvolutionModelMultiB class instance.
        """
        # Initialize the super-class:
        SparseDeconvolutionModel.__init__(self,
                                          data,
                                          bvecs,
                                          bvals,
                                          solver=solver,
                                          solver_params=solver_params,
                                          params_file=params_file,
                                          axial_diffusivity=axial_diffusivity,
                                          radial_diffusivity=radial_diffusivity,
                                          affine=affine,
                                          mask=mask,
                                          scaling_factor=scaling_factor,
                                          sub_sample=sub_sample,
                                          over_sample=over_sample,
                                          mode=mode,
                                          verbose=verbose)
                                              
        # Separate b values and grab the indices and values:
        bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
              
        if 0 in unique_b:
            self.b0_list = bval_list[0];
            self.b0_inds = b_inds[0];
            ind = 1
        else:
            ind = 0
        
        self.bval_list = bval_list[ind:]
        self.b_inds = b_inds[ind:]
        self.unique_b = unique_b[ind:]
        self.all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
        self.rounded_bvals = rounded_bvals
        
        if over_sample is None:
            #self.rot_vecs = np.squeeze(self.bvecs[:, self.all_b_idx])
            self.rot_vecs = self.bvecs
            
        # Name the params file, if needed: 
        this_class = str(self.__class__).split("'")[-2].split('.')[-1]
        self.params_file = params_file_resolver(self,
                                                this_class,
                                                params_file=params_file)

        # Deal with the solver stuff: 
        # For now, the default is ElasticNet:
        if solver is None:
            this_solver = sklearn_solvers['ElasticNet']
        # Assume it's a key into the dict: 
        elif isinstance(solver, str):
            this_solver = sklearn_solvers[solver]
        # Assume it's a class: 
        else:
            this_solver = solver
        
        # This will be passed as kwarg to the solver initialization:
        if solver_params is None:
            # This seems to be good for our data:
            alpha = 0.0005
            l1_ratio = 0.6
            self.solver_params = dict(alpha=alpha,
                                      l1_ratio=l1_ratio,
                                      fit_intercept=True,
                                      positive=True)
        else:
            self.solver_params = solver_params

        # We reuse the same class instance in all voxels: 
        self.solver = this_solver(**self.solver_params)
        
    @desc.auto_attr
    def response_function(self):
        """
        Canonical tensors that describes the presumed response of different b values
        """
        
        tensor_out = list()
        for idx in np.arange(len(self.unique_b)):
            bvecs = self.bvecs[:,self.b_inds[idx]]
            tensor_out.append(ozt.Tensor(np.diag([self.ad[idx], self.rd[idx], self.rd[idx]]), bvecs, self.bval_list[idx]))
        
        return tensor_out
        
    def _calc_rotations(self, vertices, mode=None, over_sample=None):
        """
        Given the rot_vecs of the object and a set of vertices (for the fitting
        these are the b-vectors of the measurement), calculate the rotations to
        be used as a design matrix

        """
        # unless we ask to change it, just use the mode of the object
        if mode is None:
            mode = self.mode
        
        # We will use the eigen-value/vectors from the response function
        # and rotate them around to each one of these vectors, calculating
        # the predicted signal in the bvecs of the actual measurement (even
        # when over-sampling):

        # If we have as many vertices as b-vectors, we can take the
        # b-values from the measurement
        if vertices.shape[0] == len(self.all_b_idx): 
            bvals = self.rounded_bvals
        
        eval_list = list()
        evec_list = list()
        
        for b_idx in np.arange(len(self.unique_b)):
            evals, evecs = self.response_function[b_idx].decompose
            eval_list.append(evals)
            evec_list.append(evecs)
            
        out_list = list()
        for bi in np.arange(len(self.unique_b)):
            temp_out = np.empty((self.rot_vecs[:,self.all_b_idx].shape[-1], vertices[:,self.b_inds[bi]].shape[-1]))
            this_b_inds = self.b_inds[bi]
            for idx, bvec in enumerate(self.rot_vecs[:,self.b_inds[bi]].T):
                this_rot = ozt.rotate_to_vector(bvec, eval_list[bi], evec_list[bi], vertices[:,this_b_inds], self.rounded_bvals[:,this_b_inds])
                pred_sig = this_rot.predicted_signal(1) 
                if mode == 'distance':
                    # This is the special case where we use the diffusion distance
                    # calculation, instead of the predicted signal:
                    temp_out[idx] = this_rot.diffusion_distance
                elif mode == 'ADC':
                    # This is another special case, calculating the ADC instead of
                    # using the predicted signal: 
                    temp_out[idx] = this_rot.ADC
                # Otherwise, we do one of these with the predicted signal: 
                elif mode == 'signal_attenuation':
                    # Fit to 1 - S/S0 
                    temp_out[idx] = 1 - pred_sig
                elif mode == 'relative_signal':
                    # Fit to S/S0 using the predicted diffusion attenuated signal:
                    temp_out[idx] = pred_sig
                elif mode == 'normalize':
                    # Normalize your regressors to have a maximum of 1:
                    temp_out[idx] = pred_sig / np.max(pred_sig)
                elif mode == 'log':
                    # Take the log and divide out the b value:
                    temp_out[idx] = np.log(pred_sig)
            out_list.append(temp_out)
            
        return out_list

    @desc.auto_attr
    def rotations(self):
        """
        These are the canonical tensors pointing in the direction of each of
        the bvecs in the sampling scheme. If an over-sample number was
        provided, we use the camino points to make canonical tensors pointing
        in all these directions (over-sampling the sphere above the resolution
        of the measurement). 
        """
        return self._calc_rotations(self.bvecs)
        
    #@desc.auto_attr
    #def b_idx(self):
        #"""
        #The indices into non-zero b values
        #"""
        #return np.where(self.bvals > 0)[0]
        
    #@desc.auto_attr
    #def b0_idx(self):
        #"""
        #The indices into zero b values
        #"""
        #return np.where(self.bvals==0)[0]
    # self.b_inds, self.b0_inds

    @desc.auto_attr
    def S0(self):
        """
        Extract and average the signal for volumes in which no b weighting was
        used (b0 scans)
        """
        return np.mean(self.data[...,self.b0_inds],-1)
        
    @desc.auto_attr
    def signal(self):
        """
        The signal in b-weighted volumes
        """
        signal_list = list()
        for si in np.arange(len(self.unique_b)):
            signal_list.append(self.data[...,self.b_inds[si]])
            
        return signal_list
        
    @desc.auto_attr
    def relative_signal(self):
        """
        The signal in each b-weighted volume, relative to the mean
        of the non b-weighted volumes
        """
        # Need to broadcast for this to work:
        signal_rel = self.signal/np.reshape(self.S0, (self.S0.shape + (1,)))
        # Convert infs to nans:
        signal_rel[np.isinf(signal_rel)] = np.nan
        
        return signal_rel

    def _flat_relative_signal(self,bi):
        """
        Get the flat relative signal only in the mask
        """       
        return np.reshape(self.relative_signal[bi,self.mask],
                         (-1, self.b_inds[bi].shape[0]))


    @desc.auto_attr
    def signal_attenuation(self):
        """
        The amount of attenuation of the signal. This is simply: 

           1-relative_signal 

        """
        return 1 - self.relative_signal

    def _flat_signal_attenuation(self,bi):
        """

        """
        return 1-self._flat_relative_signal(bi)
        
    @desc.auto_attr
    def regressors(self):
        """
        Compute the regressors and the signal to fit to, depending on the mode
        you are using  
        """
        
        iso_pred_sig = list()
        iso_regressor_list = list()
        fit_to_list = list()
        for idx, b in enumerate(self.unique_b):   
            iso_pred_sig.append(np.exp(-b * self.iso_diffusivity[idx]))
            if self.mode == 'signal_attenuation':
                iso_regressor = 1 - iso_pred_sig[idx] * np.ones(self.rotations[idx].shape[-1])
                fit_to = self._flat_signal_attenuation(idx).T
            elif self.mode == 'relative_signal':
                iso_regressor = iso_pred_sig[idx] * np.ones(self.rotations[idx].shape[-1])
                fit_to = self._flat_relative_signal(idx).T
            elif self.mode == 'normalize':
                # The only difference between this and the above is that the
                # iso_regressor is here set to all 1's, which can affect the
                # weights... 
                iso_regressor = np.ones(self.rotations[idx].shape[-1])
                fit_to = self._flat_relative_signal[idx].T
            elif self.mode == 'log':
                iso_regressor = (np.log(iso_pred_sig[idx]) *
                                np.ones(self.rotations[idx].shape[-1]))
                fit_to = np.log(self._flat_relative_signal[idx].T)
            fit_to_list.append(fit_to)
            iso_regressor_list.append(iso_regressor)
            
        # The tensor regressor always looks the same regardless of mode: 
        tensor_regressor_list = self.rotations

        return [iso_regressor_list, tensor_regressor_list, fit_to_list]
        
    def _fit_it(self, fit_to_mp, design_matrix):
        """
        The core fitting routine
        """
        # Fit the deviations from the mean of the fitted signal: 
        sig = fit_to_mp - np.mean(fit_to_mp)
        # Use the solver you created upon initialization:
        return self.solver.fit(design_matrix, sig).coef_
        
    @desc.auto_attr
    def model_params(self):
        """

        Use sklearn to fit the parameters:

        """
        # The file already exists: 
        if os.path.isfile(self.params_file):
            if self.verbose:
                print("Loading params from file: %s"%self.params_file)
            # Get the cached values and be done with it:
            return ni.load(self.params_file).get_data()

        else:

            if self.verbose:
                print("Fitting SparseDeconvolutionModel:")
                prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]

            iso_regressor_list, tensor_regressor_list, fit_to_list = self.regressors
            
            tensor_regressor_mp = np.array(tensor_regressor_list)
            fit_to_mp = np.array(fit_to_list)
            iso_regressor_list = np.array(iso_regressor_list)
            
            if self._n_vox==1:
                # We have to be a bit (too) clever here, so that the indexing
                # below works out:
                fit_to_mp = np.array([fit_to_mp]).T
                
            # We fit the deviations from the mean signal, which is why we also
            # demean each of the basis functions:
            
            design_matrix = tensor_regressor_mp - np.mean(tensor_regressor_mp, 0)

            # One basis function per column (instead of rows):
            design_matrix = design_matrix.T

            # One weight for each rotation
            params = np.empty((self._n_vox, self.rotations.shape[0]))
                
            for vox in xrange(self._n_vox):
                # Call out to the core fitting routine: 
                params[vox] = self._fit_it(fit_to.T[vox], design_matrix)
                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)

                    
            out_params = ozu.nans((self.signal.shape[:3] + 
                                        (design_matrix.shape[-1],)))
            
            out_params[self.mask] = params
            # Save the params to a file: 
            params_ni = ni.Nifti1Image(out_params, self.affine)
            if self.params_file != 'temp':
                if self.verbose:
                    print("Saving params to file: %s"%self.params_file)
                params_ni.to_filename(self.params_file)

            # And return the params for current use:
            return out_params
            
    @desc.auto_attr    
    def _flat_params(self):
        """
        Sometimes its useful to have a flat version of the params
        """
        return self.model_params[self.mask].squeeze()


    @desc.auto_attr
    def fit(self):
        """
        Predict the data from the fit of the SparseDeconvolutionModel
        """
        if self.verbose:
            msg = "Predicting signal from SparseDeconvolutionModel"
            msg += " with %s"%self.solver
            print(msg)
        
        iso_regressor, tensor_regressor, fit_to = self.regressors

        design_matrix = tensor_regressor - np.mean(tensor_regressor, 0)
        design_matrix = design_matrix.T
        out_flat = np.empty(self._flat_signal.shape)
        
        for vox in xrange(self._n_vox):
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0             
            if self.mode == 'log':
                this_relative=np.exp(np.dot(this_params, design_matrix.T)+
                                     np.mean(fit_to.T[vox]))
            else:     
                this_relative = (np.dot(this_params, design_matrix.T) + 
                                 np.mean(fit_to.T[vox]))
            if (self.mode == 'relative_signal' or self.mode=='normalize' or
                self.mode=='log'):
                this_pred_sig = this_relative * self._flat_S0[vox]
            elif self.mode == 'signal_attenuation':
                this_pred_sig =  (1 - this_relative) * self._flat_S0[vox]

            # Fit scale and offset:
            #a,b = np.polyfit(this_pred_sig, self._flat_signal[vox], 1)
            # out_flat[vox] = a*this_pred_sig + b
            out_flat[vox] = this_pred_sig 
        out = ozu.nans(self.signal.shape)
        out[self.mask] = out_flat

        return out