"""

Sparse spherical deconvolution

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

import osmosis.utils as ozu
import osmosis.descriptors as desc
import osmosis.cluster as ozc

from osmosis.model.canonical_tensor import CanonicalTensorModel, AD, RD
from osmosis.model.base import SCALE_FACTOR
from osmosis.model.io import params_file_resolver

import osmosis.model.dti as dti


class SparseDeconvolutionModel(CanonicalTensorModel):
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
                 verbose=True,
                 force_recompute=False):
        """
        Initialize SparseDeconvolutionModel class instance.
        """
        # Initialize the super-class:
        CanonicalTensorModel.__init__(self,
                                      data,
                                      bvecs,
                                      bvals,
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

        # This is only here for now, but should be implemented in the
        # base-class (all the way up?) and generalized in a wrapper to model
        # params, I believe. 
        self.force_recompute = force_recompute

    def _fit_it(self, fit_to, design_matrix):
        """
        The core fitting routine
        """
        # Fit the deviations from the mean of the fitted signal: 
        sig = fit_to - np.mean(fit_to)
        # Use the solver you created upon initialization:
        return self.solver.fit(design_matrix, sig).coef_

    @desc.auto_attr
    def design_matrix(self):
        """
        Abstract the design matrix out
        """
        # We fit the deviations from the mean signal, so we demean each of the
        # basis functions and we transpose, so that we have the  regressors on
        # columns, instead of on the rows (which is how they are generated): 
        return self.rotations.T - np.mean(self.rotations, -1)


    @desc.auto_attr
    def model_params(self):
        """

        Use sklearn to fit the parameters:

        """
        # The file already exists: 
        if os.path.isfile(self.params_file) and not self.force_recompute:
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

            iso_regressor, tensor_regressor, fit_to = self.regressors

            if self._n_vox==1:
                # We have to be a bit (too) clever here, so that the indexing
                # below works out:
                fit_to = np.array([fit_to]).T

            # One weight for each rotation
            params = np.empty((self._n_vox, self.rotations.shape[0]))
                
            for vox in xrange(self._n_vox):
                # Call out to the core fitting routine: 
                params[vox] = self._fit_it(fit_to.T[vox], self.design_matrix)
                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)
                    
            out_params = ozu.nans((self.signal.shape[:3] + 
                                        (self.design_matrix.shape[-1],)))
            
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
        out_flat = np.empty(self._flat_signal.shape)
        
        for vox in xrange(self._n_vox):
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0             
            if self.mode == 'log':
                this_relative=np.exp(np.dot(self.design_matrix, this_params)+
                                     np.mean(fit_to.T[vox]))
            else:     
                this_relative = (np.dot(self.design_matrix, this_params) + 
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


    def predict(self, vertices):
        """
        Predict the signal on a new set of vertices
        """
        if self.verbose:
            msg = "Predicting signal from SparseDeconvolutionModel"
            msg += " with %s"%self.solver
            print(msg)

        # For this one, we need a different design matrix, which we calculate
        # here now:
        design_matrix = self._calc_rotations(vertices)
        design_matrix = design_matrix.T - np.mean(design_matrix, -1)
        
        iso_regressor, tensor_regressor, fit_to = self.regressors
        out_flat = np.empty((self._flat_signal.shape[0], vertices.shape[-1]))
        for vox in xrange(out_flat.shape[0]):
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0 
            if self.mode == 'log':
                this_relative=np.exp(np.dot(design_matrix, this_params)+
                                     np.mean(fit_to.T[vox]))
            else:     
                this_relative = (np.dot(design_matrix, this_params) + 
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

        out = ozu.nans(self.signal.shape[:3]+ (vertices.shape[-1],))
        out[self.mask] = out_flat

        return out


    @desc.auto_attr
    def fit_angle(self):
        """
        The angle between the tensors that were fitted
        """
        out_flat = np.empty(self._flat_signal.shape[0])
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(self._flat_params[vox][0]):
                idx1 = np.argsort(self._flat_params[vox])[-1]
                idx2 = np.argsort(self._flat_params[vox])[-2]
                ang = np.rad2deg(ozu.vector_angle(
                    self.bvecs[:,self.b_idx].T[idx1],
                    self.bvecs[:,self.b_idx].T[idx2]))

                ang = np.min([ang, 180-ang])
                
                out_flat[vox] = ang
                
        else:
            out_flat[vox] = np.nan
        
        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = out_flat

        return out

    @desc.auto_attr
    def odf_peaks(self):
        """
        Calculate the value of the peaks in the ODF (in this case, that is
        defined as the weights on the model params 
        """
        faces = dps.Sphere(xyz=self.bvecs[:,self.b_idx].T).faces
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
        out_flat = ozu.nans(self._flat_signal.shape[0])
        flat_odf_peaks = self.odf_peaks[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_odf_peaks[vox][0]):
                idx1 = np.argsort(flat_odf_peaks[vox])[-1]
                idx2 = np.argsort(flat_odf_peaks[vox])[-2]
                if idx1 != idx2:
                    ang = np.rad2deg(ozu.vector_angle(
                        self.bvecs[:,self.b_idx].T[idx1],
                        self.bvecs[:,self.b_idx].T[idx2]))

                    ang = np.min([ang, 180-ang])
                
                    out_flat[vox] = ang
                        
        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = out_flat
        return out
        

    def n_peaks(self, threshold=0.1):
        """
        How many peaks in the ODF of each voxel
        """
        return np.sum(self.odf_peaks > threshold, -1)


    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        Gives you not only the principal, but also the 2nd, 3rd, etc
        """
        out_flat = ozu.nans(self._flat_signal.shape + (3,))
        # flat_peaks = self.odf_peaks[self.mask]
        flat_peaks = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            coeff_idx = np.where(flat_peaks[vox]>0)[0]
            for i, idx in enumerate(coeff_idx):
                out_flat[vox, i] = self.bvecs[:,self.b_idx].T[idx]
        
        out = ozu.nans(self.signal.shape + (3,))
        out[self.mask] = out_flat
            
        return out
        
        
    def quantitative_anisotropy(self, Np):
        """
        Return the relative size and indices of the Np major param values
        (canonical tensor weights) in the ODF 
        """
        if self.verbose:
            print("Calculating quantitative anisotropy:")
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]


        # Allocate space for Np QA values and indices in the entire volume:
        qa_flat = np.zeros((self._flat_params.shape[0], Np))
        inds_flat = np.zeros(qa_flat.shape, np.int)  # indices! 
        
        for vox in xrange(self._flat_params.shape[0]):
            this_params = self._flat_params[vox]
            ii = np.argsort(this_params)[::-1]  # From largest to smallest
            inds_flat[vox] = ii[:Np]
            qa_flat[vox] = (this_params/np.sum(this_params))[inds_flat[vox]] 

            if self.verbose:
                prog_bar.animate(vox, f_name=f_name)

        qa = np.zeros(self.signal.shape[:3] + (Np,))
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
        di = ozu.nans(self.data.shape[:3])
        di_flat = np.zeros(self._n_vox)
        for vox in xrange(self._n_vox):
            inds = np.argsort(self._flat_params[vox])[::-1] # From largest to
                                                            # smallest
            nonzero_idx = np.where(self._flat_params[vox][inds]>0)
            if len(nonzero_idx[0])>0:
                # Only look at the non-zero weights:
                vox_idx = inds[nonzero_idx].astype(int)
                this_mp = self._flat_params[vox][vox_idx]
                this_dirs = self.rot_vecs.T[vox_idx]
                n_idx = len(vox_idx)
                if all_to_all:
                    di_s = np.zeros(n_idx)
                    # Calculate this as all-to-all:
                    angles = np.arccos(np.dot(this_dirs, this_dirs.T))
                    for ii in xrange(n_idx):
                        this_di_s = 0 
                        for jj in  xrange(ii+1, n_idx): 
                            ang = angles[ii, jj]
                            di_s[ii] += np.sin(ang) * ((this_mp[ii]*this_mp[jj])/
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
                                          np.sin(angles))

        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = di_flat
        return out

        
    def anisotropy_index(self):
        """
        We calculate an anisotropy index according to the following:

        .. math:
        
           AI = \sum_{i=1}^{n}{w_i}/\sum_{i=0}^{n}{w_i}

        Where the 0th index refers to the isotropic weight, defined here to be
        the mean of $\frac{S}{S_0}$.
       
        """
        raise NotImplementedError

    
    @desc.auto_attr
    def cluster_fodf(self, in_data=None):
        """
        Use k-means clustering to find the peaks in the fodf

        Per default, we'll use AIC to determine the value of `k`. However, if
        an additional data-set is provided, we will use the prediction of this
        additional data as a criterion for stopping. Once additional k stops
        improving cross-validation accuracy, that's a good time to stop.

        
        """
        centroid_arr = np.empty(len(self._flat_signal), dtype=object)

        # If you provided another object that inherits from DWI,  
        if in_data:
            comp_data = in_data.data[self.mask]
        
        for vox in range(len(self._flat_signal)):
            this_fodf = self._flat_params[vox]
            # Find the bvecs for which the parameters are non-zero:
            nz_idx = np.where(this_fodf>0)

            # If there's nothing here, just give it the origin and move on: 
            if len(nz_idx[0]) == 0:
                centroid_arr[vox] = np.array([0, 0, 0])
                break

            # Get them in the right orientation and shape:
            bv = self.bvecs[:, self.b_idx].T[nz_idx].T
            
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
        out = np.ones(self.signal.shape[:3], dtype=object) * np.nan
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
            vertices = self.bvecs[:, self.b_idx]

        design_matrix = self._calc_rotations(vertices, mode=mode)
        
        out_flat = np.empty((self._flat_signal.shape[0], vertices.shape[-1]))
        for vox in xrange(out_flat.shape[0]):
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0 
            out_flat[vox] = np.dot(this_params, design_matrix.T)
            
        out = ozu.nans(self.signal.shape[:3]+ (vertices.shape[-1],))
        out[self.mask] = out_flat
        return out


    @desc.auto_attr
    def model_adc(self):
        """
        
        """
        fit_rel_sig = self.fit[self.mask]/self._flat_S0.reshape(self._n_vox,1)
        log_rel_sig = np.log(fit_rel_sig)

        out_flat = log_rel_sig/(-self.bvals[self.b_idx][0])
        out = ozu.nans(self.signal.shape)
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
        bD = np.exp(self.bvals[:,self.b_idx][0]* 3.0)
        mu = np.mean(self.regressors[1])
        beta0 = np.empty(s_bar.shape)
        for vox in xrange(beta0.shape[-1]): 
            beta0[vox] = (s_bar[vox] - mu * np.sum(self._flat_params[vox])) * bD

        
        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = beta0

        return out

    def odf(self, sphere, interp_kwargs=dict(function='multiquadric', smooth=0)):
        """
        Interpolate the fiber odf into a provided sphere class instance (from
        dipy)
        """
        s0 = dps.Sphere(xyz=self.bvecs[:, self.b_idx].T)
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
class SparseDeconvolutionFitter(object):
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
        iso_regressor, tensor_regressor, _ = self.cache.regressors

        design_matrix = tensor_regressor - np.mean(tensor_regressor, 0)
        fit_to = data[self.cache.b_idx]/np.mean(data[self.cache.b0_idx])        
        self.cache.model_params = self.cache._fit_it(fit_to, design_matrix)
            
        return self.cache



