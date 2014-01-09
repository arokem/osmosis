"""

Sparse spherical deconvolution

"""


import os
import inspect
import warnings

import numpy as np
from scipy.optimize import nnls
import scipy.optimize as opt
# Get stuff from sklearn, if that's available:
try:
    from sklearn.linear_model import Lasso, LassoCV
    # Get other stuff from sklearn.linear_model:
    from sklearn.linear_model import ElasticNet, Lars, Ridge, ElasticNetCV, LinearRegression
    # Get OMP:
    from sklearn.linear_model.omp import OrthogonalMatchingPursuit as OMP

    has_sklearn = True

    # Make a dict with solvers to be used for choosing among them:
    sklearn_solvers = dict(Lasso=Lasso,
                           OMP=OMP,
                           ElasticNet=ElasticNet,
                           ElasticNetCV=ElasticNetCV,
                           Lars=Lars,
                           LR=LinearRegression,
                           nnls=nnls)

except ImportError:
    e_s = "Could not import sklearn. Download and install from XXX"
    warnings.warn(e_s)
    has_sklearn = False    


import nibabel as ni
import dipy.reconst.recspeed as recspeed
import dipy.core.sphere as dps
import dipy.core.geometry as geo
import dipy.data as dpd

import osmosis.utils as ozu
import osmosis.descriptors as desc
import osmosis.cluster as ozc
import osmosis.tensor as ozt
import osmosis.mean_diffusivity_models as mdm
import osmosis.leastsqbound as lsq
from osmosis.utils import separate_bvals

import osmosis.model.dti as dti
from osmosis.model.canonical_tensor import CanonicalTensorModel, AD, RD

# from osmosis.model.base import SCALE_FACTOR
from osmosis.model.io import params_file_resolver


SCALE_FACTOR = 1000.0 


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
        
        if solver is "LR":
            self.solver = this_solver(None)
        elif solver is "nnls":
            self.solver = this_solver
        else:
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

class SparseDeconvolutionModelMultiB(SparseDeconvolutionModel):
    """
    Sparse spherical deconvolution of diffusion data with multiple b values.
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 solver=None,
                 solver_params=None,
                 initial = None,
                 bounds = None,
                 mean = "mean_model",
                 params_file=None,
                 axial_diffusivity=AD, # Should be a dict
                 radial_diffusivity=RD, # Should be a dict
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 mode='relative_signal',
                 verbose=True,
                 fit_method = "LS"):
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
        (self.bval_list_rm0, self.b_inds_rm0,
         self.unique_b_rm0, self.rounded_bvals_rm0) = separate_bvals(bvals, mode = 'remove0')
         
        if 0 in bvals:
            self.all_b_idx = np.squeeze(np.where(bvals != 0))
            self.b0_inds = np.squeeze(np.where(bvals == 0))
        else:
            self.all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
            self.b0_list = bval_list[0]
            self.b0_inds = b_inds[0]
            
        self.b_idx = self.all_b_idx
        self.rounded_bvals = rounded_bvals
        self.unique_b = unique_b[1:]
        
        # Name the params file, if needed: 
        this_class = str(self.__class__).split("'")[-2].split('.')[-1]
        self.params_file = params_file_resolver(self,
                                                this_class,
                                                params_file=params_file)
        if over_sample is None:
            self.rot_vecs = bvecs[:, self.all_b_idx]
        elif np.logical_and(isinstance(over_sample, int), over_sample<len(self.bvals[self.all_b_idx])):
            self.rot_vecs = ozu.get_camino_pts(over_sample)
        elif over_sample in[362, 642]:
            # We want to get these vertices:
            verts = dpd.get_sphere('symmetric%s'%over_sample).vertices
            self.rot_vecs = verts.T
            
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
        if solver is "LR":
            self.solver = this_solver(None)
        elif solver is "nnls":
            self.solver = this_solver
        else:
            self.solver = this_solver(**self.solver_params)
        
        # Keep the solver as a string for conveniance:
        self.solver_str = solver
        
        # Model of the means
        self.func = mdm.two_decaying_exp_plus_const
        self.mean = mean
        self.fit_method = fit_method
        
        # Restraints on the parameters for fitting the mean model
        if bounds is None:
            # Use default restraints for semi bi-exponential mean model
            self.bounds = [(-10000, 0), (-10000, 0), (-10000, 0), (-10000, 0)]
        else:
            self.bounds = bounds
        
        # Initial values for fitting the mean model
        if initial is None:
            # Use default initial values for semi bi-exponential mean model
            self.initial = (-0.6, -1, -0.75, -0.3)
        else:
            self.initial = initial

        
    def response_function(self, bval_tensor, vertices, bvals=None):
        """
        Canonical tensors that describes the presumed response of different b values
        
        Parameters
        ----------
        bval_tensor: int
            B value of the current vertex not scaled by the scaling factor
        vertices: 2 dimensional array
            Vertices to find the canonical tensor to.
        
        Returns
        -------
        tensor_out: object
            Diffusion tensor object for extraction of eigenvalues and eignvectors later
        """
        if bvals is None:
            bvals = np.array([bval_tensor])
            
        tensor_out = ozt.Tensor(np.diag([self.ad[bval_tensor], self.rd[bval_tensor],
                                self.rd[bval_tensor]]), vertices, bvals)
        
        return tensor_out
        
    def _calc_rotations(self, vertices, bvals, b_idx=None, mode=None, over_sample=None):
        """
        Given the rot_vecs of the object and a set of vertices (for the fitting
        these are the b-vectors of the measurement), calculate the rotations to
        be used as a design matrix
        
        Parameters
        ----------
        bvals: 1 dimensional array
            B values scaled by the scaling factor
        vertices: 2 dimensional array
            B vectors
        
        Returns
        -------
        out: 1 dimensional array
            Response function at these particular b vectors
        """
        
        # unless we ask to change it, just use the mode of the object
        if mode is None:
            mode = self.mode
        
        # We will use the eigen-value/vectors from the response function
        # and rotate them around to each one of these vectors, calculating
        # the predicted signal in the bvecs of the actual measurement (even
        # when over-sampling):
        
        if len(vertices.shape) == 1:
            vertex = np.reshape(vertices, (3,1))
        
        if self.mean == "empirical":
            bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bvals)
            [evals, evecs, these_verts,
            these_bvals, out] = self._calc_rotations_empirical(bvals, b_inds, vertices, b_idx)
        else:
            # Here, bvals is just one b value divided by the scaling factor
            these_bvals = np.array([bvals])
            these_verts = vertices
            out = np.empty((self.rot_vecs.shape[-1], vertices.shape[-1])) 
            bval_tensor = round(bvals)*self.scaling_factor
            evals, evecs = self.response_function(bval_tensor, these_verts).decompose
        
        # bvec within the rotational vectors
        for idx, bvec in enumerate(self.rot_vecs.T):               
            # these_bvals needs to be divided by the scaling factor before this operation:
            this_rot = ozt.rotate_to_vector(bvec, evals, evecs, these_verts, these_bvals)
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
    
    def _calc_rotations_empirical(self, bval_arr, b_inds, vertices, b_idx):
        """
        Helper function for _calc_rotations only used of demeaning by the empirical
        mean.
        
        Parameters
        ----------
        bval_arr: 1 dimensional array
            B values scaled by the scaling factor
        b_inds: list
            List of indices corresponding to each b value
        vertices: 2 dimensional array
            B vectors
        b_idx: int
            Current index into b_inds
        
        Returns
        -------
        evals: 1 dimensional array
            Eigenvalues from the response function
        evecs: 2 dimensional array
            Eigenvectors from the response function
        these_verts: 2 dimensional array
            Reduced b vectors for current b value
        these_bvals: 1 dimensional array
            Reduced b values for current b value
        out: 2 dimensional array
            Preallocated array to hold response function at these particular b vectors
        """
        # bval_arr comes in without a scaling factor, comes out with a scaling factor
        bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(bval_arr)
        if 0 in unique_b:
            ind = 1
        else:
            # For predict function.  Input b values don't usually include b = 0 values
            ind = 0
        bval_list = bval_list[ind:]
        unique_b = unique_b[ind:]
        all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
        
        if len(unique_b) > 1:
            b_inds = b_inds[ind:]
            this_b_inds = b_inds[b_idx]
            these_verts = vertices[:, this_b_inds]
            out = np.empty((self.rot_vecs.shape[-1], vertices[:,this_b_inds].shape[-1]))
            these_bvals = np.squeeze(rounded_bvals)[this_b_inds]/self.scaling_factor
        else:
            if ind == 1:
                # If b = 0 values included in input b value array, then take only the
                # non-b=0 values and vectors.
                this_b_inds = b_inds[ind]
                these_verts = vertices[:, this_b_inds]
                1/0.
                these_bvals = rounded_bvals[this_b_inds]/self.scaling_factor
                out = np.empty((self.rot_vecs.shape[-1], vertices[:, this_b_inds].shape[-1]))
            else:
                # Otherwise, the input values are already the non-b=0 values.
                these_verts = vertices
                these_bvals = rounded_bvals/self.scaling_factor
                out = np.empty((self.rot_vecs.shape[-1], vertices.shape[-1]))
        
        bval_tensor = int(self.unique_b[b_idx]) # Not divided by scaling factor
        evals, evecs = self.response_function(bval_tensor, these_verts, bvals=these_bvals).decompose
        
        return evals, evecs, these_verts, these_bvals, out
        
    def rotations(self, b_idx):
        """
        These are the canonical tensors pointing in the direction of each of
        the bvecs in the sampling scheme. If an over-sample number was
        provided, we use the camino points to make canonical tensors pointing
        in all these directions (over-sampling the sphere above the resolution
        of the measurement). 
        """
        return self._calc_rotations(self.bvecs, self.rounded_bvals, b_idx=b_idx)
    
    @desc.auto_attr
    def tensor_model(self):
        """
        Create a tensor model object in order to get the mean diffusivities
        
        Returns
        -------
        Tensor object for extracting mean diffusivities later
        """
        these_bvals = self.bvals*self.scaling_factor
        these_bvals[self.b0_inds] = 0
        return dti.TensorModel(self.data, self.bvecs, these_bvals, mask=self.mask,
                                params_file='temp')

    def _flat_MD_rel_sig_avg(self, bvals, idx, md = None):
        """
        Compute the relative signal average for demeaning of the signal.
        
        Parameters
        ----------
        bvals: 1 dimensional array
            B values at which to evaluate the mean diffusivity at
        idx: int
            Index into the b values for the current b value
        md: 1 dimensional array
            Input mean diffusivities if the default mean diffusivity calculated from
            the SFM's b values and b vectors is not desired.
            
        Return
        ------
        out: 1 dimensional array
            Relative signal calculated from mean diffusivity
        """
        if md is None:
            out = np.exp(-bvals[idx]*self.tensor_model.mean_diffusivity[self.mask])
        else:
            out = np.exp(-bvals[idx]*md)

        return out

    def _flat_rel_sig_avg(self, bvals):
        """
        Compute the relative signal average for demeaning of the signal.
        
        Parameters
        ----------
        bvals: 1 dimensional array
            B values at which to evaluate the mean model at
        
        Returns
        -------
        sig_out: 1 dimensional array
            Means for each b value in each direction
        params_out: 2 dimensional array
            Parameters for the mean model at each voxel
        """
        flat_data = self.data[np.where(self.mask)]
        
        param_num = len(inspect.getargspec(self.func)[0])-1
        params_out = np.zeros((int(np.sum(self.mask)), param_num))
        sig_out = ozu.nans((int(np.sum(self.mask)),) + (len(self.all_b_idx),))
        
        for vox in np.arange(np.sum(self.mask)).astype(int):
            s0 = np.mean(flat_data[vox, self.b0_inds], -1)
            s_prime = np.log(flat_data[vox, self.all_b_idx]/s0)
            
            params, _ = opt.leastsq(mdm.err_func, self.initial, args=(bvals, s_prime, self.func))
            
            lsq_b_out = lsq.leastsqbound(mdm.err_func, self.initial,
                                                args=(bvals, s_prime, self.func),
                                                bounds = self.bounds)
            params = lsq_b_out[0]                                             
            params_out[vox] = np.squeeze(params)
            sig_out[vox] = np.exp(self.func(bvals, *params))        
        
        return sig_out, params_out
        
    @desc.auto_attr
    def fit_flat_rel_sig_avg(self):
        """
        The relative signal average of the data
        
        Returns
        -------
        Relative signal average calculated using the mean model at the SFM's
        default b values.
        """
        return self._flat_rel_sig_avg(self.bvals[self.all_b_idx])
                                                               
    @desc.auto_attr                  
    def regressors(self):
        """
        Compute the regressors and the signal to fit to, depending on the mode
        you are using
        
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
        
        fit_to = np.empty((np.sum(self.mask), len(self.all_b_idx)))
        fit_to_means = np.empty((np.sum(self.mask), len(self.all_b_idx)))
        fit_to_demeaned = np.empty(fit_to.shape)
        
        n_columns = len(self.rot_vecs[0])
        if self.mean == "no_demean":
            n_columns = n_columns + 1
        tensor_regressor = np.empty((len(self.all_b_idx), n_columns))
        design_matrix = np.empty(tensor_regressor.shape)

        for idx, b_idx in enumerate(self.all_b_idx):
            
            if self.mean == "MD":
                sig_demean = self._flat_MD_rel_sig_avg(self.bvals, b_idx)
            else:
                sig_out, _ = self.fit_flat_rel_sig_avg
                sig_demean = sig_out[:, idx]
                
            if self.mode == 'signal_attenuation':
                sig_avg = 1 - np.copy(sig_demean)
                this_fit_to = self._flat_signal_attenuation[:, idx]
            elif self.mode == 'relative_signal':
                sig_avg = np.copy(sig_demean)
                this_fit_to = self._flat_relative_signal[:, idx]
            elif self.mode == 'normalize':
                # The only difference between this and the above is that the
                # iso_regressor is here set to all 1's, which can affect the
                # weights...
                sig_avg = np.copy(sig_demean)
                this_fit_to = self._flat_relative_signal[:, idx]
            elif self.mode == 'log':
                #sig_avg = np.log(np.copy(sig_demean))
                this_fit_to = np.log(self._flat_relative_signal[:, idx])
            
            # Find tensor regressor values
            this_tensor_regressor = self._calc_rotations(
                                        np.reshape(self.bvecs[:, b_idx], (3,1)), self.bvals[b_idx])
            if self.mean == "no_demean":
                tensor_regressor[idx] = np.concatenate((np.squeeze(this_tensor_regressor),[1]))
            else:
                tensor_regressor[idx] = np.squeeze(this_tensor_regressor)
            
            # Find the signals to fit to and demean them by mean signal calculated from
            # the mean diffusivity.
            fit_to[:, idx] = this_fit_to
            fit_to_demeaned[:, idx] = this_fit_to - sig_avg
            fit_to_means[:, idx] = sig_avg

            if self.mean == "MD":
                bval_tensor = round(self.bvals[b_idx])*self.scaling_factor
                this_MD = (self.ad[bval_tensor]+2*self.rd[bval_tensor])/3.
                design_matrix[idx] = (np.squeeze(this_tensor_regressor) -
                                        np.exp(-self.bvals[b_idx]*this_MD))
        if self.mean == "MD":
            return [fit_to, tensor_regressor, fit_to_demeaned, fit_to_means, design_matrix]
        else:
            return [fit_to, tensor_regressor, fit_to_demeaned, fit_to_means]
            
    @desc.auto_attr                  
    def empirical_regressors(self):
        """
        Compute the regressors and the signal to fit to, depending on the mode
        you are using.  This is only used when demeaning by the empirical mean.
        
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
        
        fit_to = np.empty((np.sum(self.mask), len(self.all_b_idx)))
        fit_to_means = np.empty((np.sum(self.mask), len(self.all_b_idx)))
        fit_to_demeaned = np.empty(fit_to.shape)
        
        n_columns = len(self.rot_vecs[0])
        tensor_regressor = np.empty((len(self.all_b_idx), n_columns))
        design_matrix = np.empty(tensor_regressor.shape)
        
        for idx, b in enumerate(self.unique_b):
            if len(self.unique_b) > 1:
                flat_sig_inds = self.b_inds_rm0[idx]
            else:
                flat_sig_inds = self.b_inds_rm0
                
            if self.mode == 'signal_attenuation':
                this_fit_to = self._flat_signal_attenuation[:, flat_sig_inds].T
            elif self.mode == 'relative_signal':
                this_fit_to = self._flat_relative_signal[:, flat_sig_inds].T
            elif self.mode == 'normalize':
                # The only difference between this and the above is that the
                # iso_regressor is here set to all 1's, which can affect the
                # weights... 
                this_fit_to = self._flat_relative_signal[:, flat_sig_inds].T
            elif self.mode == 'log':
                this_fit_to = np.log(self._flat_relative_signal(flat_sig_inds)).T
            
            this_tensor_regressor = self.rotations(idx)

            for vox in xrange(self._n_vox):                
                # Tensor regressors
                tensor_regressor[flat_sig_inds] = this_tensor_regressor.T
                
                # Array of signals to fit to - Means only, demeaned, and normal
                fit_to[vox, flat_sig_inds] = this_fit_to.T[vox]
                fit_to_demeaned[vox, flat_sig_inds] = this_fit_to.T[vox] - np.mean(this_fit_to.T[vox])
                fit_to_means[vox, flat_sig_inds] = np.mean(this_fit_to.T[vox])
                
                # Design matrix - tensor regressors with mean subtracted
                this_design_matrix = this_tensor_regressor.T - np.mean(this_tensor_regressor, -1)
                design_matrix[flat_sig_inds] = this_design_matrix
                
        return [fit_to, tensor_regressor, fit_to_demeaned, fit_to_means, design_matrix]
    
    def _fit_it(self, fit_to, design_matrix, solver_str):
        """
        The core fitting routine
        
        Parameters
        ----------
        fit_to: 2 dimensional array
            Values to fit to and demeaned by some kind of mean at each voxel
            and at each direction
        design_matrix: 2 dimensional array
            Demeaned design matrix for fitting
        
        Returns
        -------
        Solution found using input solver.  Weights on each of the columns
        of the design matrix for a particular voxel.
        """
        # Use the solver you created upon initialization:
        if solver_str == "nnls":
            # nnls solver is not a part of the sklearn solvers and doesn't have .coef_
            # attribute.  Also, the output of the fit_it comes out as a tuple rather
            # than array where the first entry is the solution.
            return self.solver(design_matrix, fit_to)[0]
        else:
            return self.solver.fit(design_matrix, fit_to).coef_
                   
    def _flat_signal_b(self, b_inds):
        """
        Get the signal in the diffusion-weighted volumes in flattened form
        (only in the mask).
        
        Parameters
        ----------
        b_inds: list
            List of indices corresponding to each b value
            
        Returns
        -------
        Signal in the diffusion-weighted volumes in flattened form
        """
        flat_sig = self._flat_data[:,b_inds]
            
        return flat_sig
        
    @desc.auto_attr
    def model_params(self):
        """

        Use sklearn to fit the parameters:
        
        Returns
        -------
        Solution found using input solver.  Weights on each of the columns
        of the design matrix at each voxel.
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
                prog_bar = ozu.ProgressBar(self._flat_signal_b(self.all_b_idx).shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]
            
            if self.mean == "MD":
                _, _, fit_to, _, design_matrix = self.regressors
            elif self.mean == "empirical":
                _, _, fit_to, _, design_matrix  = self.empirical_regressors
            else:
                sig_out, _ = self.fit_flat_rel_sig_avg
                fit_to_with_mean, tensor_regressor, fit_to, _ = self.regressors
            
            if self._n_vox==1:
                # We have to be a bit (too) clever here, so that the indexing
                # below works out:
                fit_to_with_mean = fit_to_with_mean.T
                       
            # One weight for each rotation
            col_num = self.rot_vecs.shape[-1]
            if self.mean == "no_demean":
                col_num = col_num + 1
            params = np.empty((self._n_vox, col_num))
                           
            for vox in xrange(self._n_vox):
                if np.logical_or(self.mean == "MD", self.mean == "empirical"):
                    vox_fit_to_demeaned = fit_to[vox]
                else:
                    if self.mean == "mean_model":
                        avg_sig = sig_out[vox][:, None]
                    elif self.mean == "no_demean":
                        avg_sig = 0.0
                    design_matrix = tensor_regressor - avg_sig
                    vox_fit_to_demeaned = fit_to_with_mean[vox] - np.squeeze(avg_sig)
                    if self.fit_method == "WLS":
                        sig_out = sig_out.astype(float)
                        weighting_matrix = np.diag(sig_out[vox]/np.max(sig_out[vox]))
                        design_matrix = np.dot(weighting_matrix, design_matrix)
                        vox_fit_to_demeaned = np.dot(weighting_matrix, vox_fit_to_demeaned)                   
                    
                params[vox] = self._fit_it(vox_fit_to_demeaned, design_matrix, self.solver_str)
                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)
            
            # It doesn't matter what's in the last dimension since we only care
            # about the first 3.  Thus, just pick the array of signals from them
            # first b value.
            out_params = ozu.nans(self.signal.shape[:3] + (design_matrix.shape[-1],))
            
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
        
        if self.mean == "MD":
            _,_,_,fit_to_means, design_matrix = self.regressors
        elif self.mean == "empirical":
            _,_,_,fit_to_means, design_matrix = self.empirical_regressors
        else:
            _, tensor_regressor, _, fit_to_means = self.regressors
            sig_out, _ = self.fit_flat_rel_sig_avg
        
        out_flat_arr = np.zeros(fit_to_means.shape)
        for vox in xrange(self._n_vox):
            if np.logical_and(self.mean != "MD", self.mean != "empirical"):
                design_matrix = tensor_regressor - sig_out[vox][:, None]
                
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0
            
            if self.mode == 'log':
                this_relative=np.exp(np.dot(this_params, design_matrix.T) +
                                     fit_to_means[vox])
            else:     
                this_relative = np.dot(this_params, design_matrix.T) + fit_to_means[vox]
            if (self.mode == 'relative_signal' or self.mode=='normalize' or
                self.mode=='log'):
                this_pred_sig = this_relative * self._flat_S0[vox] # this_relative = S/S0
            elif self.mode == 'signal_attenuation':
                this_pred_sig =  (1 - this_relative) * self._flat_S0[vox]

            out_flat_arr[vox] = this_pred_sig
            
        out = ozu.nans((self.signal.shape[:3] + 
                         (design_matrix.shape[-1],)))
        out[self.mask] = out_flat_arr

        return out
        
    def predict(self, vertices, new_bvals, new_params = None, md = None):
        """
        Predict the signal on a new set of vertices
        
        Parameters
        ----------
        vertices: 2 dimensional array
            New b vectors to predict data at
        new_bvals: 1 dimensional array
            Corresponding b values for the new b vectors
        new_params: 2 dimensional array
            New parameters for a mean model not calculated through the default
            b values and b vectors for the SFM class
        md: 1 dimensional array
            New mean diffusivities not calculated through the default b values and
            b vectors for the SFM class
        
        Returns
        -------
        out: 4 dimensional array
            Volume of predicted values at each input b vector
        """

        if self.verbose:
            msg = "Predicting signal from SparseDeconvolutionModel"
            msg += " with %s"%self.solver
            print(msg)
        
        if self.mean == "empirical":
            out_flat_arr, fit_to_mean, design_matrix = self._empirical_predict(new_bvals, vertices)
        else:
            # Just so everything works out, divide by the scaling factor
            new_bvals = new_bvals/self.scaling_factor
            
            if len(vertices.shape) == 1:
                vertices = np.reshape(vertices, (3,1))
            
            fit_to_means = np.mean(self.regressors[0], -1)
            
            col_num = self.rot_vecs.shape[-1]
            if self.mean == "no_demean":
                col_num = col_num + 1

            tensor_regressor = np.zeros((vertices.shape[-1], col_num))
            design_matrix = np.zeros((vertices.shape[-1], self.rot_vecs.shape[-1]))
            fit_to_mean = np.zeros((self._n_vox, vertices.shape[-1])) # For MD only
               
            for idx, bval in enumerate(new_bvals):
                # Create a new design matrix from the given vertices
                cr = self._calc_rotations(np.reshape(vertices[:, idx], (3,1)), bval)
                if self.mean == "no_demean":
                    tensor_regressor[idx] = np.concatenate((np.squeeze(cr), [1]))
                else:
                    this_tensor_regressor = np.squeeze(cr)
                    tensor_regressor[idx] = this_tensor_regressor
                    
                if self.mean == "MD":
                    bval_tensor = round(bval)*self.scaling_factor
                    this_MD = (self.ad[bval_tensor]+2*self.rd[bval_tensor])/3.
                    design_matrix[idx] = np.squeeze(this_tensor_regressor) - np.exp(-bval*this_MD)
                    
                    # Find the mean signal across the vertices corresponding to the b values
                    # given.
                    fit_to_mean[:, idx] = self._flat_MD_rel_sig_avg(new_bvals, idx, md = md)

            # If new parameters are given, use those instead.
            if new_params is not None:
                params_out = new_params
            else:
                # Grab the parameters for fitting the mean
                _, params_out = self.fit_flat_rel_sig_avg
            
            out_flat_arr = np.zeros((self._n_vox, vertices.shape[-1]))
        
        # Now that everything is set up, predict the signal in the given vertices.
        for vox in xrange(self._n_vox):    
            this_params = self._flat_params[vox]
            this_params[np.isnan(this_params)] = 0.0

            if np.logical_or(self.mean == "MD", self.mean == "empirical"):
                this_fit_to_mean = fit_to_mean[vox]
            else:
                this_fit_to_mean = np.exp(self.func(new_bvals, *params_out[vox]))
                if self.mean == "mean_model":
                    this_fit_to_mean = this_fit_to_mean[:, None]
                elif self.mean == "no_demean":
                    this_fit_to_mean = 0.0
                design_matrix = tensor_regressor - this_fit_to_mean
            # Relative signal:
            if self.mode == 'log':
                this_relative=np.exp(np.dot(this_params, design_matrix.T) +
                                    np.squeeze(this_fit_to_mean))
            else:
                this_relative = (np.dot(this_params, design_matrix.T) +
                                    np.squeeze(this_fit_to_mean))
            # Predicted signal        
            if (self.mode == 'relative_signal' or self.mode=='normalize' or
                self.mode=='log'):
                this_pred_sig = this_relative * self._flat_S0[vox] # this_relative = S/S0
            elif self.mode == 'signal_attenuation':
                this_pred_sig =  (1 - this_relative) * self._flat_S0[vox]
                
            out_flat_arr[vox] = this_pred_sig
        
        out = ozu.nans(self.data.shape[:3] + (out_flat_arr.shape[-1],))
        out[self.mask] = out_flat_arr
        
        return out
        
    def _empirical_predict(self, new_bvals, vertices):
        """
        Helper function for predict.  Only used if demeaning by the empirical mean.
        
        Parameters
        ----------
        new_bvals: 1 dimensional array
            Corresponding b values for the new b vectors
        vertices: 2 dimensional array
            New b vectors to predict data at
        
        Returns
        -------
        out_flat_arr: 2 dimensional array
            Preallocated array for storing predicted values at each input b vector
        fit_to_mean: 2 dimensional array
            Mean across the vertices of each b value at each voxel
        design_matrix: 2 dimensional array
            Demeaned design matrix for fitting
        """
        bval_list, b_inds, unique_b, rounded_bvals = separate_bvals(new_bvals)
        [bval_list_rm0, b_inds_rm0,
        unique_b_rm0, rounded_bvals_rm0] = separate_bvals(new_bvals, mode = 'remove0')
        
        design_matrix = np.zeros((vertices.shape[-1], self.rot_vecs.shape[-1]))
        for mpi in np.arange(len(unique_b)):
            tensor_regressor = self._calc_rotations(vertices, new_bvals, b_idx=mpi)
            this_design_matrix = tensor_regressor.T - np.mean(tensor_regressor, -1)
            if len(unique_b) == 1:
                design_matrix[b_inds_rm0] = np.squeeze(this_design_matrix)
            else:
                design_matrix[b_inds_rm0[mpi]] = this_design_matrix
            
        fit_to, _, _, _, _ = self.empirical_regressors
        
        # Find the mean signal across the vertices corresponding to the b values
        # given.
        fit_to_mean = np.zeros((fit_to.shape[0], vertices.shape[-1]))
        for vox in xrange(self._n_vox):
            if len(unique_b) == 1:
                fit_to_mean[vox, b_inds_rm0] = np.mean(fit_to[vox, self.b_inds_rm0])
            else:
                for b_fi in np.arange(len(unique_b)):
                    idx = np.squeeze(np.where(self.unique_b == unique_b[b_fi]))
                    fit_to_mean[vox, b_inds_rm0[b_fi]] = np.mean(fit_to[vox, self.b_inds_rm0[idx]])
        
        out_flat_arr = np.zeros(np.squeeze(fit_to_mean).shape)
        
        return out_flat_arr, fit_to_mean, design_matrix
