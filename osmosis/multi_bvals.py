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
from osmosis.utils import separate_bvals

from osmosis.model.sparse_deconvolution import SparseDeconvolutionModel
import osmosis.model.dti as dti
from osmosis.model.canonical_tensor import AD, RD

# from osmosis.model.base import SCALE_FACTOR
from osmosis.model.io import params_file_resolver


SCALE_FACTOR = 1000.0 


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
                 axial_diffusivity=AD, # Should be a dict
                 radial_diffusivity=RD, # Should be a dict
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
        self.bval_list_rm0, self.b_inds_rm0, self.unique_b_rm0, self.rounded_bvals_rm0 = separate_bvals(bvals, mode = 'remove0')
              
        if 0 in np.unique(rounded_bvals):
            self.b0_list = bval_list[0]
            self.b0_idx = b_inds[0]
            self.b0_inds = b_inds[0]
            ind = 1
        else:
            ind = 0
        
        self.bval_list = bval_list[ind:]
        self.b_inds = b_inds[ind:]
        self.unique_b = unique_b[ind:]
        self.all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
        self.b_idx = self.all_b_idx
        self.rounded_bvals = rounded_bvals
        
        # Name the params file, if needed: 
        this_class = str(self.__class__).split("'")[-2].split('.')[-1]
        self.params_file = params_file_resolver(self,
                                                this_class,
                                                params_file=params_file)
        if over_sample is None:
            self.rot_vecs = bvecs[:, self.all_b_idx]
        elif np.logical_and(isinstance(over_sample, int), over_sample<len(self.bvals[self.all_b_idx])):
            self.rot_vecs = ozu.get_camino_pts(over_sample)
            
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
        
    def response_function(self, bval_tensor, vertex):
        """
        Canonical tensors that describes the presumed response of different b values
        """
        tensor_out = ozt.Tensor(np.diag([self.ad[bval_tensor], self.rd[bval_tensor],
                                self.rd[bval_tensor]]), vertex, np.array([bval_tensor]))
        
        return tensor_out
        
    def _calc_rotations(self, bval, vertex, mode=None, over_sample=None):
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
        
        if len(vertex.shape) == 1:
            vertex = np.reshape(vertex, (3,1))
            
        out = np.empty((self.rot_vecs.shape[-1], vertex.shape[-1]))
        
        bval_tensor = round(bval)*self.scaling_factor
        evals, evecs = self.response_function(bval_tensor, vertex).decompose
      
        for idx, bvec in enumerate(self.rot_vecs.T):               
            # bvec within the rotational vectors
            this_rot = ozt.rotate_to_vector(bvec, evals, evecs, vertex, np.array([bval]))
            pred_sig = this_rot.predicted_signal(1)
            
            #Try saving some memory: 
            del this_rot
            
            if mode == 'distance':
                    # This is the special case where we use the diffusion distance
                    # calculation, instead of the predicted signal:
                    out[idx] = this_rot.diffusion_distance
            elif mode == 'ADC':
                    # This is another special case, calculating the ADC instead of
                    # using the predicted signal: 
                    out[idx] = this_rot.ADC
            # Otherwise, we do one of these with the predicted signal: 
            elif mode == 'signal_attenuation':
                    # Fit to 1 - S/S0 
                    out[idx] = 1 - pred_sig
            elif mode == 'relative_signal':
                    # Fit to S/S0 using the predicted diffusion attenuated signal:
                    out[idx] = pred_sig
            elif mode == 'normalize':
                    # Normalize your regressors to have a maximum of 1:
                    out[idx] = pred_sig / np.max(pred_sig)
            elif mode == 'log':
                    # Take the log and divide out the b value:
                    out[idx] = np.log(pred_sig)
                       
        return out
        
    @desc.auto_attr
    def _tensor_model(self):
        """
        Create a tensor model object in order to get the mean diffusivities
        """
        these_bvals = self.bvals*self.scaling_factor
        these_bvals[self.b0_inds] = 0
        return dti.TensorModel(self.data, self.bvecs, these_bvals, mask=self.mask,
                                params_file='temp')
    
    def _flat_rel_sig_avg(self, bvals, idx, md = "None", b_mode = "None"):
        """
        Compute the relative signal average for demeaning of the signal.
        """
        if b_mode is "None":
            out = np.exp(-bvals[idx]*self._tensor_model.mean_diffusivity[self.mask])
        elif b_mode is "across_b":
            out = np.exp(-bvals[idx]*md)
        
        return out
                                                               
    @desc.auto_attr                  
    def regressors(self):
        """
        Compute the regressors and the signal to fit to, depending on the mode
        you are using  
        """
        
        fit_to = np.empty((np.sum(self.mask), len(self.all_b_idx)))
        fit_to_means = np.empty((np.sum(self.mask), len(self.all_b_idx)))
        fit_to_demeaned = np.empty(fit_to.shape)
        n_columns = len(self.rot_vecs[0,:])
        tensor_regressor = np.empty((len(self.all_b_idx), n_columns))
        design_matrix = np.empty(tensor_regressor.shape)
        
        for idx, b_idx in enumerate(self.all_b_idx):
            if self.mode == 'signal_attenuation':
                sig_avg = 1 - self._flat_rel_sig_avg(self.bvals, b_idx)
                this_fit_to = self._flat_signal_attenuation[:, idx]
            elif self.mode == 'relative_signal':
                sig_avg = self._flat_rel_sig_avg(self.bvals, b_idx)
                this_fit_to = self._flat_relative_signal[:, idx]
            elif self.mode == 'normalize':
                # The only difference between this and the above is that the
                # iso_regressor is here set to all 1's, which can affect the
                # weights...
                sig_avg = self._flat_rel_sig_avg(self.bvals, b_idx)
                this_fit_to = self._flat_relative_signal[:, idx]
            elif self.mode == 'log':
                sig_avg = np.log(self._flat_rel_sig_avg)(self.bvals, b_idx)
                this_fit_to = np.log(self._flat_relative_signal[:, idx])
            
            # Find tensor regressor values and demean by the theoretical average signal
            this_tensor_regressor = self._calc_rotations(self.bvals[b_idx],
                                        np.reshape(self.bvecs[:, b_idx], (3,1)))
            bval_tensor = round(self.bvals[b_idx])*self.scaling_factor
            this_MD = (self.ad[bval_tensor]+2*self.rd[bval_tensor])/3.
            tensor_regressor[idx] = np.squeeze(this_tensor_regressor)
            design_matrix[idx] = np.squeeze(this_tensor_regressor) - np.exp(-self.bvals[b_idx]*this_MD)
            
            # Find the signals to fit to and demean them by mean signal calculated from
            # the mean diffusivity.
            fit_to[:, idx] = this_fit_to
            fit_to_demeaned[:, idx] = this_fit_to - sig_avg
            fit_to_means[:, idx] = sig_avg

        return [fit_to, tensor_regressor, design_matrix, fit_to_demeaned, fit_to_means]
        
    def _fit_it(self, fit_to, design_matrix):
        """
        The core fitting routine
        """
        # Use the solver you created upon initialization:
        return self.solver.fit(design_matrix, fit_to).coef_
        
    @desc.auto_attr
    def _n_vox(self):
        """
        This is the number of voxels in the masked region.
        
        Used mainly to differentiate the single-voxel case from the multi-voxel
        case.  
        """
        # We must be prepared to deal with single-voxel case: 
        if len(self._flat_signal_b(self.b_inds[0]).shape)==1:
            return 1
        # Otherwise, we are going to assume this is a 2D thing, once we
        # have flattened it: 
        else:
            return self._flat_signal_b(self.b_inds[0]).shape[0]
            
    def _flat_signal_b(self, b_inds):
        """
        Get the signal in the diffusion-weighted volumes in flattened form
        (only in the mask).
        """
        flat_sig = self._flat_data[:,b_inds]
            
        return flat_sig
        
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
                prog_bar = ozu.ProgressBar(self._flat_signal_b(self.b_inds[0]).shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]
            
            _,_ , design_matrix, fit_to, _  = self.regressors
            
            if self._n_vox==1:
                # We have to be a bit (too) clever here, so that the indexing
                # below works out:
                this_fit_to = fit_to.T
            else:
                this_fit_to = fit_to
                       
            # One weight for each rotation
            params = np.empty((self._n_vox, self.rot_vecs.shape[-1]))
            
            for vox in xrange(self._n_vox):
                params[vox] = self._fit_it(this_fit_to[vox], design_matrix)
                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)
            
            # It doesn't matter what's in the last dimension since we only care
            # about the first 3.  Thus, just pick the array of signals from them
            # first b value.
            out_params = ozu.nans(self.signal.shape[:3] + 
                                  (design_matrix.shape[-1],))
            
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
    def _flat_S0(self):
        """
        Get the signal in the b0 scans in flattened form (only in the mask)
        """
        return np.mean(self._flat_data[:,self.b0_inds], -1)
        
    @desc.auto_attr
    def fit(self):
        """
        Predict the data from the fit of the SparseDeconvolutionModel
        """
        if self.verbose:
            msg = "Predicting signal from SparseDeconvolutionModel"
            msg += " with %s"%self.solver
            print(msg)
        
        _, _, design_matrix, _, fit_to_means = self.regressors
        
        out_flat_arr = np.zeros(fit_to_means.shape)
        for vox in xrange(self._n_vox):    
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0
            
            if self.mode == 'log':
                this_relative=np.exp(np.dot(this_params, design_matrix.T) +
                                    self.fit_to_means[vox])
            else:     
                this_relative = np.dot(this_params, design_matrix.T) + fit_to_means[vox]
            if (self.mode == 'relative_signal' or self.mode=='normalize' or
                self.mode=='log'):
                this_pred_sig = this_relative * self._flat_S0[vox] # this_relative = S/S0
            elif self.mode == 'signal_attenuation':
                this_pred_sig =  (1 - this_relative) * self._flat_S0[vox]

            # Fit scale and offset:
            #a,b = np.polyfit(this_pred_sig, self._flat_signal[vox], 1)
            # out_flat[vox] = a*this_pred_sig + b
            out_flat_arr[vox] = this_pred_sig
            
        out = ozu.nans((self.signal.shape[:3] + 
                         (design_matrix.shape[-1],)))
        out[self.mask] = out_flat_arr

        return out
        
    def predict(self, vertices, new_bvals, md = "None", b_mode = "None"):
        """
        Predict the signal on a new set of vertices
        """
        if self.verbose:
            msg = "Predicting signal from SparseDeconvolutionModel"
            msg += " with %s"%self.solver
            print(msg)
        
        # Just so everything works out:
        new_bvals = new_bvals/1000.
        
        if len(vertices.shape) == 1:
            vertices = np.reshape(vertices, (3,1))
        
        design_matrix = np.zeros((vertices.shape[-1], self.rot_vecs.shape[-1]))
        fit_to_mean = np.zeros((self._n_vox, vertices.shape[-1]))
        for idx, bval in enumerate(new_bvals):
            # Create a new design matrix from the given vertices
            tensor_regressor = self._calc_rotations(bval, np.reshape(vertices[:, idx], (3,1)))
            bval_tensor = round(bval)*self.scaling_factor
            this_MD = (self.ad[bval_tensor]+2*self.rd[bval_tensor])/3.
            design_matrix[idx] = np.squeeze(tensor_regressor) - np.exp(-bval*this_MD)
            
            # Find the mean signal across the vertices corresponding to the b values
            # given.
            fit_to_mean[:, idx] = self._flat_rel_sig_avg(new_bvals, idx, md = md, b_mode = b_mode)

        out_flat_arr = np.zeros(np.squeeze(fit_to_mean).shape)
        
        # Now that everthing is set up, predict the signal in the given vertices.
        for vox in xrange(self._n_vox):    
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0
            
            if self.mode == 'log':
                this_relative=np.exp(np.dot(this_params, design_matrix.T) +
                                    fit_to_mean[vox])
            else:     
                this_relative = (np.dot(this_params, design_matrix.T) +
                                    fit_to_mean[vox])
            if (self.mode == 'relative_signal' or self.mode=='normalize' or
                self.mode=='log'):
                this_pred_sig = this_relative * self._flat_S0[vox] # this_relative = S/S0
            elif self.mode == 'signal_attenuation':
                this_pred_sig =  (1 - this_relative) * self._flat_S0[vox]
                
            out_flat_arr[vox] = this_pred_sig
           
        out = ozu.nans(self.data.shape[:3] + (out_flat_arr.shape[-1],))
        out[self.mask] = out_flat_arr

        return out
        
    #def predict_n(data, bvals, bvecs, mask, n, b_mode, AD = "None", RD = "None"):
        #"""
        #Predicts signals for a certain percentage of the vertices.
        
        #Parameters
        #----------
        #data: 4 dimensional array
            #Diffusion MRI data
        #bvals: 1 dimensional array
            #All b values
        #bvecs: 3 dimensional array
            #All the b vectors
        #mask: 3 dimensional array
            #Brain mask of the data
        #n: int
            #Integer indicating the percent of vertices that you want to predict
        #b_mode: str
            #'all': if fitting to all b values
            #'bvals': if fitting to individual b values
            
        #Returns
        #-------
        #actual: 2 dimensional array
            #Actual signals for the predicted vertices
        #predicted: 2 dimensional array 
            #Predicted signals for the vertices left out of the fit
        #""" 
        #bval_list, b_inds, unique_b, rounded_bvals = snr.separate_bvals(bvals)
        #_, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = snr.separate_bvals(bvals,
                                                                            #mode = 'remove0')
        #all_b_idx = np.squeeze(np.where(rounded_bvals != 0))    
        
        #actual = np.empty((np.sum(mask), len(all_b_idx)))
        #predicted = np.empty(actual.shape)
        
        ## Generate the regressors in the full model from which we choose the regressors in
        ## the reduced model from.  This is so you won't have 
        #if b_mode == 'all': 
            #full_mod = sfm_mb.SparseDeconvolutionModelMultiB(data, bvecs, bvals,
                                                            #mask = mask,
                                                            #params_file = "temp")                 
        #for bi in np.arange(len(unique_b[1:])):
            
            #if b_mode is "all":
                #all_inc_0 = np.arange(len(rounded_bvals))
                #bvals_pool = bvals
            #elif b_mode is "bvals":
                #all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][bi]))
                #bvals_pool = rounded_bvals
                #if np.logical_and(AD is not "None", RD is not "None"):
                    #this_AD = AD[bi]
                    #this_RD = RD[bi]
            
            #these_b_inds = b_inds[1:][bi]
            #these_b_inds_rm0 = b_inds_rm0[bi]
            #vec_pool = np.arange(len(these_b_inds))
            
            ## Need to choose random indices so shuffle them!
            #np.random.shuffle(vec_pool)
            
            ## How many of the indices are you going to leave out at a time?
            #num_choose = np.ceil((n/100.)*len(these_b_inds))
                    
            #for combo_num in np.arange(np.floor(100./n)):
                #these_inc0 = list(all_inc_0)
                #idx = list(b_inds_rm0[bi])
                #low = (combo_num)*num_choose
                #high = np.min([(combo_num*num_choose + num_choose), len(vec_pool)])
                #vec_pool_inds = vec_pool[low:high]
                
                #vec_combo = these_b_inds[vec_pool_inds]
                #vec_combo_rm0 = these_b_inds_rm0[vec_pool_inds]
                
                ## Remove the chosen indices from the rest of the indices
                #for choice_idx in vec_pool_inds:
                    #these_inc0.remove(these_b_inds[choice_idx])
                    #idx.remove(these_b_inds_rm0[choice_idx])
                
                #for b_idx in np.arange(len(unique_b[1:])):
                    #if np.logical_and(b_idx != bi, b_mode is "all"):
                        #idx = np.concatenate((idx, b_inds_rm0[b_idx]),0)
                    
                ## Make the list back into an array
                #these_inc0 = np.array(these_inc0)
                
                ## Isolate the b vectors, b values, and data not including those to be predicted
                #these_bvecs = bvecs[:, these_inc0]
                #these_bvals = bvals_pool[these_inc0]
                #this_data = data[:, :, :, these_inc0]
                
                ## Need to sort the indices first before indexing full model's regressors
                #si = sorted(idx)
                
                #if b_mode is "all":
                    #mod = sfm_mb.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                                #mask = mask, params_file = "temp")
                    #if np.logical_and(AD is not "None", RD is not "None"):
                        #mod = sfm_mb.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                                      #mask = mask, params_file = "temp",
                                                                                #axial_diffusivity = AD,
                                                                               #radial_diffusivity = RD)
                    ## Grab regressors from full model's preloaded regressors
                    #fit_to = full_mod.regressors[0][:, si]
                    #tensor_regressor = full_mod.regressors[1][:, si][si, :]
                    #mod.regressors = demean(fit_to, tensor_regressor, mod)
                                
                    #predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo])

                #elif b_mode is "bvals":
                    #mod = sfm.SparseDeconvolutionModel(this_data, these_bvecs, these_bvals,
                                                        #mask = mask, params_file = "temp")
                    #if np.logical_and(AD is not "None", RD is not "None"):    
                        #mod = sfm.SparseDeconvolutionModel(this_data, these_bvecs, these_bvals,
                                                            #mask = mask, params_file = "temp",
                                                                  #axial_diffusivity = this_AD,
                                                                 #radial_diffusivity = this_RD)
                    #predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo])[mod.mask]
                #actual[:, vec_combo_rm0] = data[mod.mask][:, vec_combo]
               
        #return actual, predicted
        
    #def demean(fit_to, tensor_regressor, mod):
        #"""
        #This function demeans the signals and tensor regressors.
        
        #Parameters
        #----------
        #fit_to: 2 dimensional array
            #Original signal fitted to.  Size should be equal to the number of voxels by the
            #number of directions.
        #tensor_regressor: 2 dimensional array
            #The predicted signals from the tensor model.  Size should be equal to the number
            #of directions fit to by the number of directions fit to.
            
        #Returns
        #-------
        #fit_to: 2 dimensional array
            #Original signal fitted to.  Size should be equal to the number of voxels by the
            #number of directions.
        #tensor_regressor: 2 dimensional array
            #The predicted signals from the tensor model.  Size should be equal to the number
            #of directions fit to by the number of directions fit to.
        #design_matrix: 2 dimensional array
            #Demeaned tensor regressors
        #fit_to_demeaned: 2 dimensional array
            #Demeaned signal fitted to
        #fit_to_means:
            #The means of the original signal fitted to.
        #"""
        
        #fit_to_demeaned = np.empty(fit_to.shape)
        #fit_to_means = np.empty(fit_to.shape)
        #design_matrix = np.empty(tensor_regressor.shape)
        
        #for bidx, b in enumerate(mod.unique_b):
            #for vox in xrange(mod._n_vox):
                ## Need to demean everything across the vertices that were fitted to
                #fit_to_demeaned[vox, mod.b_inds_rm0[bidx]] = (fit_to[vox, mod.b_inds_rm0[bidx]]
                                                    #- np.mean(fit_to[vox, mod.b_inds_rm0[bidx]]))
                #fit_to_means[vox, mod.b_inds_rm0[bidx]] = np.mean(fit_to[vox, mod.b_inds_rm0[bidx]])
                #design_matrix[mod.b_inds_rm0[bidx]] = (tensor_regressor[mod.b_inds_rm0[bidx]]
                                        #- np.mean(tensor_regressor[mod.b_inds_rm0[bidx]].T, -1))
                                        
        #return [fit_to, tensor_regressor, design_matrix, fit_to_demeaned, fit_to_means]   
        
    #def predict_bvals(data, bvals, bvecs, mask, b_fit_to, b_predict):
        #"""
        #Predict for each b value.
        
        #Parameters
        #----------
        #data: 4 dimensional array
            #Diffusion MRI data
        #bvals: 1 dimensional array
            #All b values
        #bvecs: 3 dimensional array
            #All the b vectors
        #mask: 3 dimensional array
            #Brain mask of the data
        #b_fit_to: int
            #Unique b value index of the b value to fit to.
        #b_predict: int
            #Unique b value index of the b value to predict.
            
        #Returns
        #-------
        #actual: 2 dimensional array
            #Actual signals for the predicted vertices
        #predicted: 2 dimensional array 
            #Predicted signals for the vertices left out of the fit
        #"""
        
        #bval_list, b_inds, unique_b, rounded_bvals = snr.separate_bvals(bvals)
        #bval_list_rm0, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = snr.separate_bvals(bvals,
                                                                            #mode = 'remove0')
        #all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][b_fit_to]))
            
        #mod = sfm.SparseDeconvolutionModel(data[:,:,:,all_inc_0], bvecs[:,all_inc_0],
                                                #rounded_bvals[all_inc_0], mask = mask,
                                                                    #params_file = 'temp')
        #actual = data[mod.mask][0, b_inds[1:][b_predict]]
        #predicted = mod.predict(bvecs[:, b_inds[1:][b_predict]])[mod.mask][0]
            
        #return actual, predicted

    #def nchoosek(n,k):
        #"""
        #Finds all the number of unique combinations from choosing groups of k from a pool of n.
        
        #Parameters
        #----------
        #n: int
            #Number of items in the pool you are choosing from
        #k: int
            #Size of the groups you are choosing from the pool
            
        #n!/(k!*(n-k)!)
        #"""
        #return f(n)/f(k)/f(n-k)
        
    #def choose_AD_RD(AD_start, AD_end, RD_start, RD_end, AD_num, RD_num):
        #"""
        #Parameters
        #----------
        #AD_start: int
            #Lowest axial diffusivity desired
        #AD_end: int
            #Highest axial diffusivity desired
        #RD_start: int
            #Lowest radial diffusivity desired
        #RD_end: int
            #Highest radial diffusivity desired
        #AD_num: int
            #Number of different axial diffusivities
        #RD_num: int
            #Number of different radial diffusivities
            
        #Returns
        #-------
        #AD_combos: obj
            #Unique axial diffusivity combinations
        #RD_combos: obj
            #Unique radial diffusivity combinations
        #"""
        
        #AD_bag = np.linspace(AD_start, AD_end, num = AD_num)
        #RD_bag = np.linspace(RD_start, RD_end, num = RD_num)

        #AD_combos = list(itertools.combinations(AD_bag, 3))
        #RD_combos = list(itertools.combinations(RD_bag, 3))
        
        #return AD_combos, RD_combos
        
    #def predict_RD_AD(AD_start, AD_end, RD_start, RD_end, AD_num, RD_num, data, bvals, bvecs, mask):
        #"""
        #Predicts vertices with different axial and radial diffusivities and finds them
        #root-mean-square error (rmse) between the actual values and predicted values.
        
        #Parameters
        #----------
        #AD_start: int
            #Lowest axial diffusivity desired
        #AD_end: int
            #Highest axial diffusivity desired
        #RD_start: int
            #Lowest radial diffusivity desired
        #RD_end: int
            #Highest radial diffusivity desired
        #AD_num: int
            #Number of different axial diffusivities
        #RD_num: int
            #Number of different radial diffusivities
        #data: 4 dimensional array
            #Diffusion MRI data
        #bvals: 1 dimensional array
            #All b values
        #bvecs: 3 dimensional array
            #All the b vectors
        #mask: 3 dimensional array
            #Brain mask of the data
            
        #Returns
        #-------
        #rmse_b: 1 dimensional array
            #The rmse from fitting to individual b values
        #rmse_mb: 1 dimensional array
            #The rmse from fitting to all the b values
        #AD_order: list
            #The groups of axial diffusivities in the order they were chosen
        #RD_order: list
            #The groups of radial diffusivities in the order they were chosen
        #"""
        #AD_combos, RD_combos = choose_AD_RD(AD_start, AD_end, RD_start, RD_end, AD_num, RD_num)
        
        #AD_order = []
        #RD_order = []
        #rmse_b = np.empty((np.sum(mask), nchoosek(AD_num,3)*nchoosek(RD_num,3)))
        #rmse_mb = np.empty(rmse_b.shape)

        #track = 0
        #for AD_idx in np.arange(len(AD_combos)):
            #for RD_idx in np.arange(len(RD_combos)):
                #actual_b, predicted_b = predict_n(data, bvals, bvecs, mask, np.array(AD_combos[AD_idx]), np.array(RD_combos[RD_idx]), 'bvals')
                #actual, predicted = predict_n(data, bvals, bvecs, mask, np.array(AD_combos[AD_idx]), np.array(RD_combos[RD_idx]), 'all')
                
                #rmse_b[:, track] = np.sqrt(np.mean((actual_b - predicted_b)**2, -1))
                #rmse_mb[:, track] = np.sqrt(np.mean((actual - predicted)**2, -1))
                
                #track = track + 1
                
                #AD_order.append(AD_combos[AD_idx])
                #RD_order.append(RD_combos[RD_idx])
                
        #return rmse_b, rmse_mb, AD_order, RD_order
        
    @desc.auto_attr
    def fit_angle(self):
        """
        The angle between the tensors that were fitted
        """
        out_flat = np.empty(self._flat_signal_all.shape[0])
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(self._flat_params[vox][0]):
                idx1 = np.argsort(self._flat_params[vox])[-1]
                idx2 = np.argsort(self._flat_params[vox])[-2]
                ang = np.rad2deg(ozu.vector_angle(
                    self.bvecs[:,self.all_b_idx].T[idx1],
                    self.bvecs[:,self.all_b_idx].T[idx2]))

                ang = np.min([ang, 180-ang])
                
                out_flat[vox] = ang
                
        else:
            out_flat[vox] = np.nan
        
        out = ozu.nans(self.signal[0].shape[:3])
        out[self.mask] = out_flat

        return out
        
    @desc.auto_attr
    def odf_peaks(self):
        """
        Calculate the value of the peaks in the ODF (in this case, that is
        defined as the weights on the model params 
        """
        faces = dps.Sphere(xyz=self.bvecs[:,self.all_b_idx].T).faces
        if self._n_vox == 1: 
            odf_flat = np.array([self.model_params])
        else: 
            odf_flat = self.model_params[self.mask]
        out_flat = np.zeros(odf_flat.shape)
        for vox in xrange(odf_flat.shape[0]):
            if ~np.any(np.isnan(odf_flat[vox])):
                this_odf = odf_flat[vox].copy()
                peaks, inds = recspeed.local_maxima(this_odf, faces)
                out_flat[vox][inds] = peaks 

        if self._n_vox == 1:
            return out_flat
        
        out = ozu.nans(self.model_params.shape)
        out[self.mask] = out_flat
        return out
        
    @desc.auto_attr
    def odf_peak_angles(self):
        """
        Calculate the angle between the two largest peaks in the odf peak
        distribution
        """
        out_flat = ozu.nans(self._flat_signal_all.shape[0])
        flat_odf_peaks = self.odf_peaks[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_odf_peaks[vox][0]):
                idx1 = np.argsort(flat_odf_peaks[vox])[-1]
                idx2 = np.argsort(flat_odf_peaks[vox])[-2]
                if idx1 != idx2:
                    ang = np.rad2deg(ozu.vector_angle(
                        self.bvecs[:,self.all_b_idx].T[idx1],
                        self.bvecs[:,self.all_b_idx].T[idx2]))

                    ang = np.min([ang, 180-ang])
                
                    out_flat[vox] = ang
                        
        out = ozu.nans(self.signal[0].shape[:3])
        out[self.mask] = out_flat
        return out
        
    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        Gives you not only the principal, but also the 2nd, 3rd, etc
        """
        out_flat = ozu.nans(self._flat_signal_all.shape + (3,))
        # flat_peaks = self.odf_peaks[self.mask]
        flat_peaks = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            coeff_idx = np.where(flat_peaks[vox]>0)[0]
            for i, idx in enumerate(coeff_idx):
                out_flat[vox, i] = self.bvecs[:,self.all_b_idx].T[idx]
        
        out = ozu.nans(self.signal[0].shape[:3] + (len(self.all_b_idx),) + (3,))
        out[self.mask] = out_flat
            
        return out
        
        
    def quantitative_anisotropy(self, Np):
        """
        Return the relative size and indices of the Np major param values
        (canonical tensor weights) in the ODF 
        """
        if self.verbose:
            print("Calculating quantitative anisotropy:")
            prog_bar = ozu.ProgressBar(self._flat_signal_all.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]


        # Allocate space for Np QA values and indices in the entire volume:
        qa_flat = np.zeros((self._flat_signal.shape[0], Np))
        inds_flat = np.zeros(qa_flat.shape, np.int)  # indices! 
        
        for vox in xrange(self._flat_params.shape[0]):
            this_params = self._flat_params[vox]
            ii = np.argsort(this_params)[::-1]  # From largest to smallest
            inds_flat[vox] = ii[:Np]
            qa_flat[vox] = (this_params/np.sum(this_params))[inds_flat[vox]] 

            if self.verbose:
                prog_bar.animate(vox, f_name=f_name)

        qa = np.zeros(self.signal[0].shape[:3] + (Np,))
        qa[self.mask] = qa_flat
        inds = np.zeros(qa.shape)
        inds[self.mask] = inds_flat
        return qa, inds
        
    def dispersion_index(self, all_to_all=False):
        """
        Calculate a dispersion index based on the formula:

        .. math::
        
            DI = \frac{\sum_{i=2}^{n}{\beta_i^2 alpha_i}}{\sum{i=1}{n}{\beta_i^2}}


        where $\beta_i$ is the weight in each direction, denoted by $alpha_i$,
        relative to the direction of the maximal weight.

        Or (when `all_to_all` is set to `True`)

        .. math::

           DI = \frac{\sum{i=1}^{n}\sum_{j=1}^{n}{\beta_i \beta_j alpha_ij}}{\sum{i=1}{n}{\beta_i^2}}

        where now $\alpha_i$ now denotes the angle between 
        
        """
        # Take values up to the number of measurements:
        qa, inds = self.quantitative_anisotropy(len(self.all_b_idx))
        inds_flat = inds[self.mask]
        qa_flat = qa[self.mask]

        # We'll use the original weights, not the QA for the calculation of the
        # index: 
        mp_flat = self.model_params[self.mask]
        
        di = ozu.nans(self.data.shape[:3])
        di_flat = np.zeros(self._n_vox)
        for vox in xrange(self._n_vox):
            nonzero_idx = np.where(qa_flat[vox]>0)
            if len(nonzero_idx[0])>0:
                # Only look at the non-zero weights:
                vox_idx = inds_flat[vox][nonzero_idx].astype(int)
                this_mp = mp_flat[vox][vox_idx]
                this_dirs = self.bvecs[:, self.all_b_idx].T[vox_idx]
                n_idx = len(vox_idx)
                if all_to_all:
                    di_s = np.zeros(n_idx)
                    # Calculate this as all-to-all:
                    angles = np.arccos(np.dot(this_dirs, this_dirs.T))
                    for ii in xrange(n_idx):
                        this_di_s = 0 
                        for jj in  xrange(ii+1, n_idx): 
                            ang = angles[ii, jj]
                            di_s[ii] += ang * ((this_mp[ii]*this_mp[jj])/
                                               np.sum(this_mp**2))  

                    di_flat[vox] = np.mean(di_s)/n_idx
                else:

                    #Calculate this from the highest peak to each one of the
                    #others:
                    this_pdd, dirs = this_dirs[0], this_dirs[1:] 
                    angles = np.arccos(np.dot(dirs, this_pdd))
                    angles = np.min(np.vstack([angles, np.pi-angles]), 0)
                    angles = angles/(np.pi/2)
                    di_flat[vox] = np.dot(this_mp[1:]**2/np.sum(this_mp**2),
                                          angles)

        out = ozu.nans(self.signal[0].shape[:3])
        out[self.mask] = di_flat
        return out
        
    @desc.auto_attr
    def cluster_fodf(self, in_data=None):
        """
        Use k-means clustering to find the peaks in the fodf

        Per default, we'll use AIC to determine the value of `k`. However, if
        an additional data-set is provided, we will use the prediction of this
        additional data as a criterion for stopping. Once additional k stops
        improving cross-validation accuracy, that's a good time to stop.

        
        """
        centroid_arr = np.empty(len(self._flat_signal_all), dtype=object)

        # If you provided another object that inherits from DWI,  
        if in_data:
            comp_data = in_data.data[self.mask]
        
        for vox in range(len(self._flat_signal_all)):
            this_fodf = self._flat_params[vox]
            # Find the bvecs for which the parameters are non-zero:
            nz_idx = np.where(this_fodf>0)

            # If there's nothing here, just give it the origin and move on: 
            if len(nz_idx[0]) == 0:
                centroid_arr[vox] = np.array([0, 0, 0])
                break

            # Get them in the right orientation and shape:
            bv = self.bvecs[:, self.all_b_idx].T[nz_idx].T
            
            sort_bv = bv[:, np.argsort(this_fodf[nz_idx])[::-1]]
            # We keep running k means and stop when adding more clusters stops
            # being helpful, using the BIC to calculate when to stop:
            last_bic = np.inf
            choose = np.array([0,0,0])

            # Deal with the special case of one model parameter: 
            if bv.shape[-1] == 1:
                centroids = bv * this_fodf[nz_idx]

            else: 
                for k in range(1, bv.shape[-1]):
                    # Use the k largest peaks in the data as seeds:
                    seeds = sort_bv[:, :k].T
                    centroids, y_n, sse = ozc.spkm(bv.T, k, seeds=seeds,
                                                   weights=this_fodf[nz_idx])

                    if in_data is not None:
                        # We're going to cross-validate against the other
                        # data-set: 
                        this_comp = comp_data[vox]
                        # XXX Need to do linear regression right here?           
                    else:
                        # The unexplained variance is the residual sse: 
                        bic = ozu.aic(sse, bv.shape[-1], k)

                    if bic > last_bic:
                            break
                    else:
                        choose = centroids
                        last_bic = bic
                    
            centroid_arr[vox] = centroids

        # We'll make a special nan/object array for this: 
        out = np.ones(self.signal[0].shape[:3], dtype=object) * np.nan
        out[self.mask] = centroid_arr
        return out
        
    def model_diffusion(self, vertices=None, mode='ADC'):
        """
        Calculate the ADC/diffusion distance implied by the model. This is done
        on a set of input vertices, defaulting to using the vertices of the
        measurement (the bvecs) 
        """
        # If none are provided, use the measurement points:
        if vertices is None:
            vertices = self.bvecs[:, self.all_b_idx]
        
        tensor_regressor_list = self._calc_rotations(vertices)
        
        new_design_matrix_list = list()
        for mpi in np.arange(len(self.unique_b)):
            new_design_matrix_list.append(tensor_regressor_list[mpi] - np.mean(tensor_regressor_list[mpi], 0))

        # One basis function per column (instead of rows):
        design_matrix = np.concatenate(self.design_matrix_list,-1).T
        
        out_flat = np.empty((self._flat_signal_all.shape[0], vertices.shape[-1]))
        for vox in xrange(out_flat.shape[0]):
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0 
            out_flat[vox] = np.dot(this_params, design_matrix.T)
            
        out = ozu.nans(self.signal[0].shape[:3]+ (vertices.shape[-1],))
        out[self.mask] = out_flat
        return out
        
    @desc.auto_attr
    def model_adc(self):
        """
        
        """
        fit_rel_sig = self.fit[self.mask]/self._flat_S0.reshape(self._n_vox,1)
        log_rel_sig = np.log(fit_rel_sig)

        out_flat = log_rel_sig/(-self.bvals[self.all_b_idx][0])
        out = ozu.nans(self.signal[0].shape)
        out[self.mask] = out_flat
        return out


    @desc.auto_attr
    def non_fiber_iso(self):
        """
        Calculate the part of the isotropic signal that is not due to the fiber
        component of the voxel. 
        """
        # Extract the mean signal
        s_bar = np.mean(self._flat_relative_signal, -1)
        # Take the diffusivity of water here: 
        bD = np.exp(self.bvals[:,self.all_b_idx][0]* 3.0)
        mu = np.mean(self.regressors[1])
        beta0 = np.empty(s_bar.shape)
        for vox in xrange(beta0.shape[-1]): 
            beta0[vox] = (s_bar[vox] - mu * np.sum(self._flat_params[vox])) * bD

        
        out = ozu.nans(self.signal[0].shape[:3])
        out[self.mask] = beta0

        return out

    def odf(self, sphere, interp_kwargs=dict(function='multiquadric', smooth=0)):
        """
        Interpolate the fiber odf into a provided sphere class instance (from
        dipy)
        """
        s0 = dps.Sphere(xyz=self.bvecs[:, self.all_b_idx].T)
        s1 = sphere
        params_flat = self.model_params[self.mask]
        out_flat = np.empty((self._n_vox, len(sphere.x)))
        if self._n_vox==1:
           this_params = params_flat
           this_params[np.isnan(this_params)] = 0
           out = dps.interp_rbf(this_params, s0, s1, **interp_kwargs)
           return np.squeeze(out)
        else:
            for vox in range(self._n_vox):
                this_params = params_flat[vox]
                this_params[np.isnan(this_params)] = 0
                out_flat[vox] = dps.interp_rbf(this_params, s0, s1,
                                           **interp_kwargs)
            
            out = ozu.nans(self.model_params.shape[:3] + (len(sphere.x),))
            out[self.mask] = out_flat
            return out

# The following is stuff to allow tracking with this model, using the dipy
# tracking API:
        
class SparseDeconvolutionFitter_MultiB(object):
    """
    This class conforms to the requirements of the dipy tracking API, so that
    we can use the SFM for tracking
    """
    def __init__(self,
                 gtab,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 solver_params=None,
                 params_file='temp',
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 mode='relative_signal',
                 verbose=False):
        """
        gtab : GradientTable class instance
        """
        # We initialize this with some bogus data: 
        data = np.zeros(len(gtab.bvals))
        # Make a cache with precalculated stuff
        self.cache = SparseDeconvolutionModel(data,
                                        gtab.bvecs.T,
                                        gtab.bvals,
                                        solver_params=solver_params,
                                        params_file=params_file,
                                        axial_diffusivity=axial_diffusivity,
                                        radial_diffusivity=radial_diffusivity,
                                        mask=None,
                                        # We've already scaled this mofo!
                                        scaling_factor=1,
                                        sub_sample=sub_sample,
                                        over_sample=over_sample,
                                        mode='relative_signal',
                                        verbose=verbose)
                       
        
    def fit(self, data):
        """
        Each time this is called, the data-dependent stuff gets reset. Then,
        the new data gets put in the right place, so that next time `odf` is
        triggered by the tracking API, it will apply the fitting procedure to
        this new set of data.
        """
        iso_regressor_list, tensor_regressor_list, _ = self.cache.regressors
        tensor_regressor_arr = np.concatenate(tensor_regressor_list,-1)

        design_matrix = tensor_regressor_arr.T - np.mean(tensor_regressor_arr.T, 0)
        fit_to = data[self.cache.all_b_idx]/np.mean(data[self.cache.b0_inds])        
        self.cache.model_params = self.cache._fit_it(fit_to, design_matrix)
            
        return self.cache