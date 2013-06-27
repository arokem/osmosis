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

from osmosis.model.canonical_tensor import AD, RD
from osmosis.model.base import SCALE_FACTOR
from osmosis.model.io import params_file_resolver

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
                                          verbose=verbose):
                                              
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
        self.rounded_bvals = rounded_bvals
        
        if over_sample is None:
            self.rot_vecs = self.bvecs[:,np.where(]
        
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
            tensor_out.append(ozt.Tensor(np.diag([self.ad, self.rd, self.rd]), bvecs, self.bval_list[idx]))
        
        return tensor_out
        
    @desc.auto_attr
    def regressors(self):
        """
        Compute the regressors and the signal to fit to, depending on the mode
        you are using  
        """
        bval_list, bval_ind, unique_b = separate_bvals(self.bvals)
        #b = self.bvals[self.b_idx][0]
        
        iso_pred_sig = list()
        for idx, b in enumerate(unique_b):   
            iso_pred_sig.append(np.exp(-b * self.iso_diffusivity))
        
        if self.mode == 'signal_attenuation':
            iso_regressor = 1 - iso_pred_sig * np.ones(self.rotations.shape[-1])
            fit_to = self._flat_signal_attenuation.T
        elif self.mode == 'relative_signal':
            iso_regressor = iso_pred_sig * np.ones(self.rotations.shape[-1])
            fit_to = self._flat_relative_signal.T
        elif self.mode == 'normalize':
            # The only difference between this and the above is that the
            # iso_regressor is here set to all 1's, which can affect the
            # weights... 
            iso_regressor = np.ones(self.rotations.shape[-1])
            fit_to = self._flat_relative_signal.T
        elif self.mode == 'log':
            iso_regressor = (np.log(iso_pred_sig) *
                             np.ones(self.rotations.shape[-1]))
            fit_to = np.log(self._flat_relative_signal.T)
            
        # The tensor regressor always looks the same regardless of mode: 
        tensor_regressor = self.rotations

        return [iso_regressor, tensor_regressor, fit_to]
        
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

            iso_regressor, tensor_regressor, fit_to = self.regressors

            if self._n_vox==1:
                # We have to be a bit (too) clever here, so that the indexing
                # below works out:
                fit_to = np.array([fit_to]).T
                
            # We fit the deviations from the mean signal, which is why we also
            # demean each of the basis functions:
            design_matrix = tensor_regressor - np.mean(tensor_regressor, 0)

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