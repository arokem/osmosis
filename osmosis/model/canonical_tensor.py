
import os
import inspect

import numpy as np
import scipy.optimize as opt

import nibabel as ni
import dipy.core.geometry as geo
import dipy.data as dpd

import osmosis as oz
import osmosis.utils as ozu
import osmosis.tensor as ozt
import osmosis.descriptors as desc
from osmosis.model.base import BaseModel
from osmosis.model.io import params_file_resolver
from osmosis.model.base import SCALE_FACTOR


# Global constants for this module:
AD = 1.5
RD = 0.5


class CanonicalTensorModel(BaseModel):
    """
    This is a simplified bi-tensor model, where one tensor is constrained to be a
    sphere and the other is constrained to be a canonical tensor with some
    globally set axial and radial diffusivities (e.g. based on the median ad
    and rd in the 300 highest FA voxels).

    The signal in each voxel can then be described as a linear combination of
    these two factors:

    .. math::
    
       \frac{S}{S_0} = \beta_1 e^{-bval \vec{b} q \vec{b}^t} + \beta_2 e^{-b D_0} 
 
    Where $\vec{b}$ can be chosen to be one of the diffusion-weighting
    directions used in the scan, or taken from some other collection of unit
    vectors.

    Consequently, for a particular choice of $\vec{b}$ we can write this as:

    .. math::

       S = X \beta

    Where: S is the nvoxels x ndirections $b_0$-attenuated signal from the
    entire volume, X is a 2 by ndirections matrix, with one column devoted to
    the anistropic contribution from the canonical tensor and the other column
    containing a constant term in all directions, representing the isotropic
    component. We can solve this equation using OLS fitting and derive the RMSE
    for that choice of $\vec{b}$. For each voxel, we can then find the choice
    of $\vec{b}$ that best predicts the signal (in the least-squares
    sense). This determines the estimated PDD in that voxel.
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 mode='relative_signal',
                 iso_diffusivity=None,
                 verbose=True):

        """
        Initialize a CanonicalTensorModel class instance.

        Parameters
        ----------

        params_file: str, optional
             full path to the name of the file in which to save the model
             parameters, once a model is fit. 

        over_sample: optional, int.
           Sometimes you might want to probe the sphere at a higher resolution
           than that provided by the measurement. You can do that using two
           possible sources of information. The first is the camino points,
           which ship together with osmosis and are used for the boot-strapping
           (see osmosis.boot). These are used for integers smaller than 150,
           for 246 and for 755. The other sources of information are the
           symmetric spheres provided as part of dipy. These are used if 362 or
           642 are provided. Note that these might be problematic, because they
           contain polar opposite points, so use with caution.

        mode: string, optional
        This can take one of several values, determining the form of the
        regressors and the form of the signal to fit to.

        'relative_signal': The fit is to $\frac{S}{S_0}$ and the regressors
        are the relative signal predicted for a canonical tensor and the
        relative signal predicted for a isotropic diffusivity compartment:

        .. math::

           \frac{S}{S_0} = \beta_1 e^{-bD} + \beta_2 e^{-b\vec{b}Q\vec{b}^t}

    
        'signal_attenuation': The fit is to $1-\frac{S}{S_0}$ and the
        regressors are the signal attenuation for the canonical tensor and the
        signal attenuation due to isotropic diffusion:

        .. math::

             1-\frac{S}{S_0} = \beta_1 (1-e^{-bD}) + \beta_2 (1-e^{-b\vec{b} Q \vec{b}^t})

        'normalize': in this case, we fit to $\frac{S}{S_0}$, but our regressor
        set is normalized to maximum of 1 in each columns. This affects the
        values of the weights, and also the goodness-of-fit (because of the
        relative scaling of the regressors in the OLS). The equation in
        this case is: 

        .. math::

             \frac{S}{S_0} = \beta_1 + \beta_2 \frac{e^{-b\vec{b} Q \vec{b}^t}}{max(e^{-b\vec{b} Q \vec{b}^t})}


        'log': in this case, we fit to $log(\frac{S}{S_0})$ and our regressors
        are the exponents:

        .. math::

            log(\frac{S}{S_0}) = \beta_1 -bD + \beta_1 -b\vec{b}Q\vec{b}^t

        iso_diffusivity: optional, float
            What the diffusivity of the isotropic component should be set
            to. This is irrelevant for the 'normalize' mode. Defaults to be
            equal to the axial_diffusivity
              
        """
        
        # Initialize the super-class:
        BaseModel.__init__(self,
                            data,
                            bvecs,
                            bvals,
                            affine=affine,
                            mask=mask,
                            scaling_factor=scaling_factor,
                            sub_sample=sub_sample,
                            params_file=params_file,
                            verbose=verbose)

        self.ad = axial_diffusivity
        self.rd = radial_diffusivity

        if iso_diffusivity is None:
           iso_diffusivity = axial_diffusivity
        self.iso_diffusivity = iso_diffusivity

        if over_sample is not None:
            # Symmetric spheres from dipy: 
            if over_sample in[362, 642]:
                # We want to get these vertices:
                verts = dpd.get_sphere('symmetric%s'%over_sample).vertices
                # Their convention is transposed relative to ours:
                self.rot_vecs = verts.T
            elif (isinstance(over_sample, int) and (over_sample<=150 or
                                                    over_sample in [246,755])):
                self.rot_vecs = ozu.get_camino_pts(over_sample)
            elif over_sample== 'quad132':
                pp = os.path.split(oz.__file__)[0]+'/data/SparseKernelModel/'
                self.rot_vecs = np.loadtxt(pp + 'qsph1-16-132DP.dat')[:,:-1].T
            else:
                e_s = "You asked to sample the sphere in %s"%over_sample
                e_s += " different directions. Can only do that for n<=150"
                e_s += " or n in [246, 362, 642, 755]"
                raise ValueError(e_s)
        else:
            self.rot_vecs = self.bvecs[:,self.b_idx]

        if mode not in ['relative_signal',
                        'signal_attenuation',
                        'normalize',
                        'log']:
            raise ValueError("Not a recognized mode of CanonicalTensorModel")

        self.mode = mode
        self.iso_diffusivity = iso_diffusivity
        
    @desc.auto_attr
    def response_function(self):
        """
        A canonical tensor that describes the presumed response of a single
        fiber 
        """
        bvecs = self.bvecs[:,self.b_idx]
        bvals = self.bvals[self.b_idx]
        return ozt.Tensor(np.diag([self.ad, self.rd, self.rd]), bvecs, bvals)


    def _calc_rotations(self, vertices, mode=None):
        """
        Given the rot_vecs of the object and a set of vertices (for the fitting
        these are the b-vectors of the measurement), calculate the rotations to
        be used as a design matrix

        """
        # unless we ask to change it, just use the mode of the object
        if mode is None:
            mode = self.mode

        out = np.empty((self.rot_vecs.shape[-1], vertices.shape[-1]))
        
        # We will use the eigen-value/vectors from the response function
        # and rotate them around to each one of these vectors, calculating
        # the predicted signal in the bvecs of the actual measurement (even
        # when over-sampling):

        # If we have as many vertices as b-vectors, we can take the
        # b-values from the measurement
        if vertices.shape[0] == len(self.b_idx): 
            bvals = self.bvals[self.b_idx]
        # Otherwise, we have to assume a constant b-value
        else:
            bvals = np.ones(vertices.shape[-1]) * self.bvals[self.b_idx][0]
            
        evals, evecs = self.response_function.decompose
        for idx, bvec in enumerate(self.rot_vecs.T):
            this_rot = ozt.rotate_to_vector(bvec, evals, evecs,
                                            vertices, bvals)
            pred_sig = this_rot.predicted_signal(1) 

            if mode == 'distance':
                # This is the special case where we use the diffusion distance
                # calculation, instead of the predicted signal:
                out[idx] = this_rot.diffusion_distance
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
    def rotations(self):
        """
        These are the canonical tensors pointing in the direction of each of
        the bvecs in the sampling scheme. If an over-sample number was
        provided, we use the camino points to make canonical tensors pointing
        in all these directions (over-sampling the sphere above the resolution
        of the measurement). 
        """
        return self._calc_rotations(self.bvecs[:, self.b_idx])

    @desc.auto_attr
    def regressors(self):
        """
        Compute the regressors and the signal to fit to, depending on the mode
        you are using  
        """

        b = self.bvals[self.b_idx][0]
        iso_pred_sig = np.exp(-b * self.iso_diffusivity)
        
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

        return iso_regressor, tensor_regressor, fit_to
        
    
    @desc.auto_attr
    def ols(self):
        """
        Compute the OLS solution. 
        """
        # Preallocate:
        ols_weights = np.empty((self.rotations.shape[0], 2,
                               self._flat_signal.shape[0]))

        iso_regressor, tensor_regressor, fit_to = self.regressors
        
        for idx in xrange(ols_weights.shape[0]):
            # The 'design matrix':
            d = np.vstack([tensor_regressor[idx], iso_regressor]).T
            # This is $(X' X)^{-1} X':
            ols_mat = ozu.ols_matrix(d)
            # Multiply to find the OLS solution (fitting to all the voxels in
            # one fell swoop):
            ols_weights[idx] = np.dot(ols_mat, fit_to)
            # ols_weights[idx] = npla.lstsq(d, fit_to)[0]
        return ols_weights


    @desc.auto_attr
    def model_params(self):
        """
        The model parameters.

        Similar to the TensorModel, if a fit has ocurred, the data is cached on
        disk as a nifti file 

        If a fit hasn't occured yet, calling this will trigger a model fit and
        derive the parameters.

        In that case, the steps are as follows:

        1. Perform OLS fitting on all voxels in the mask, with each of the
           $\vec{b}$. Choose only the non-negative weights. 

        2. Find the PDD that most readily explains the data (highest
           correlation coefficient between the data and the predicted signal)
           and use that one to derive the fit for that voxel

        """

        # The file already exists: 
        if os.path.isfile(self.params_file):
            if self.verbose:
                print("Loading params from file: %s"%self.params_file)

            # Get the cached values and be done with it:
            return ni.load(self.params_file).get_data()
        else:
            # Looks like we might need to do some fitting...
            # Get the bvec weights and the isotropic weights
            b_w = self.ols[:,0,:].copy().squeeze()
            i_w = self.ols[:,1,:].copy().squeeze()

            # nan out the places where weights are negative: 
            b_w[b_w<0] = np.nan
            i_w[i_w<0] = np.nan

            params = np.empty((self._flat_signal.shape[0],3))
            if self.verbose:
                print("Fitting CanonicalTensorModel:")
                prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]
            # Find the best OLS solution in each voxel:
            for vox in xrange(self._flat_signal.shape[0]):
                # We do this in each voxel (instead of all at once, which is
                # possible...) to not blow up the memory:
                vox_fits = np.empty(self.rotations.shape)
                for rot_i, rot in enumerate(self.rotations):
                    if self.mode == 'log':
                        this_sig = (np.exp(b_w[rot_i,vox] * rot +
                                    self.regressors[0][0] * i_w[rot_i,vox]) *
                                    self._flat_S0[vox])
                    else:
                        this_relative = (b_w[rot_i,vox] * rot +
                                    self.regressors[0][0] * i_w[rot_i,vox])
                        if self.mode == 'signal_attenuation':
                            this_relative = 1 - this_relative

                        this_sig = this_relative * self._flat_S0[vox]

                    vox_fits[rot_i] = this_sig
                    
                # Find the predicted signal that best matches the original
                # relative signal. That will choose the direction for the
                # tensor we use:
                corrs = ozu.coeff_of_determination(self._flat_signal[vox],
                                                   vox_fits)
                idx = np.where(corrs==np.nanmax(corrs))[0]
                
                # Sometimes there is no good solution (maybe we need to fit
                # just an isotropic to all of these?):
                if len(idx):
                    # In case more than one fits the bill, just choose the
                    # first one:
                    if len(idx)>1:
                        idx = idx[0]
                    
                    params[vox,:] = np.array([idx,
                                              b_w[idx, vox],
                                              i_w[idx, vox]]).squeeze()
                else:
                    params[vox,:] = np.array([np.nan, np.nan, np.nan])

                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)

            # Save the params for future use: 
            out_params = ozu.nans(self.signal.shape[:3] + (3,))
            out_params[self.mask] = np.array(params).squeeze()
            params_ni = ni.Nifti1Image(out_params, self.affine)
            if self.params_file != 'temp':
                if self.verbose:
                    print("Saving params to file: %s"%self.params_file)
                    params_ni.to_filename(self.params_file)

            # And return the params for current use:
            return out_params

    @desc.auto_attr
    def fit(self):
        """
        Predict the signal from the fit of the CanonicalTensorModel
        """
        if self.verbose:
            print("Predicting signal from CanonicalTensorModel")
            
        out_flat = np.empty(self._flat_signal.shape)
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_params[vox, 1]):
                if self.mode == 'log':
                    this_relative = np.exp(flat_params[vox,1] *
                                self.rotations[flat_params[vox,0]] +
                                self.regressors[0][0] * flat_params[vox,2]) 
                else: 
                    this_relative = (flat_params[vox,1] *
                                self.rotations[flat_params[vox,0]] +
                                self.regressors[0][0] * flat_params[vox,2]) 
        
                    if self.mode == 'signal_attenuation':
                        this_relative = 1 - this_relative

                out_flat[vox]= this_relative * self._flat_S0[vox]
            else:
                out_flat[vox] = np.nan
                
        out = ozu.nans(self.signal.shape)
        out[self.mask] = out_flat

        return out


    def predict(self, vertices):
        """
        Predict the signal on a novel set of vertices

        Parameters
        ----------
        vertices : an n by 3 array

        """
        # Start by generating the values of the rotations we use in these
        # coordinates on the sphere 
        rotations = self._calc_rotations(vertices)
        
        if self.verbose:
            print("Predicting signal from CanonicalTensorModel")
            
        out_flat = np.empty((self._flat_signal.shape[0], vertices.shape[-1]))
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_params[vox, 1]):
                if self.mode == 'log':
                    this_relative = np.exp(flat_params[vox,1] *
                                rotations[flat_params[vox,0]] +
                                self.regressors[0][0] * flat_params[vox,2]) 
                else: 
                    this_relative = (flat_params[vox,1] *
                                rotations[flat_params[vox,0]] +
                                self.regressors[0][0] * flat_params[vox,2]) 
        
                    if self.mode == 'signal_attenuation':
                        this_relative = 1 - this_relative

                out_flat[vox]= this_relative * self._flat_S0[vox]
            else:
                out_flat[vox] = np.nan
                
        out = ozu.nans(self.signal.shape[:3] + (vertices.shape[-1], ))
        out[self.mask] = out_flat

        return out
        
        
    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        The direction in which the best fit canonical tensor is pointing
        (in x,y,z coordinates)
        """
        flat_params = self.model_params[self.mask]
        out_flat = np.empty(flat_params.shape)

        for vox in xrange(flat_params.shape[0]):
            if not np.isnan(flat_params[vox,0]):
                out_flat[vox] = self.rot_vecs.T[flat_params[vox,0]]
            else: 
                out_flat[vox] = [np.nan, np.nan, np.nan]

        out = ozu.nans(self.model_params.shape)
        out[self.mask] = out_flat
        return out


    @desc.auto_attr
    def fractional_anisotropy(self):
        """
        An analog of the FA for this kind of model.

        .. math::

        FA = \frac{T - S}{T + S}

        Where T is the value of the tensor parameter and S is the value of the
        sphere parameter.
        """
        
        return ((self.model_params[...,1] - self.model_params[...,2])/
                (self.model_params[...,1] + self.model_params[...,2]))
    
    
class CanonicalTensorModelOpt(CanonicalTensorModel):
    """
    This one is supposed to do the same thing as CanonicalTensorModel, except
    here scipy.optimize is used to find the parameters, instead of OLS fitting.
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 mode='relative_signal',
                 iso_diffusivity=3.0,
                 model_form='flexible',
                 verbose=True):
        r"""
        Initialize a CanonicalTensorModelOpt class instance.

        Same inputs except we do not accept over-sampling, since it has no
        meaning here.

        Parameters
        ----------

        model_form: A string that chooses between different forms of the
        model. Per default, this is set to 'flexible'.

        'flexible': In this case, we fit the parameters of the following model: 

        .. math::

           \frac{S}{ S_0}= \beta_0 e^{-bD}+\beta_1 e^{-b \theta R_i Q \theta^t}

        In this model, the diffusivity of the sphere is assumed to be the
        diffusivity of water and Q is taken from the axial_diffusivity and
        radial_diffusivity inputs. We fit $\beta_0$, $\beta_1$ and $R_i$ (the
        rotation matrix, which is defined by two parameters: for the azimuth
        and elevation)
        
        'constrained': We will optimize for the following model:

        .. math::
        
           \frac{S}{ S_0}= (1-\beta) e^{-bD}+\beta e^{-b \theta R_i Q \theta^t}

        That is, a model in which the sum of the weights on isotropic and
        anisotropic components is always 1.

        'ball_and_stick': The form of the model in this case is:

        .. math::

            \frac{S}{ S_0}= (1-\beta)e^{-b d}+\beta e^{-b \theta R_idQ\theta^t}

        Note that in this case, d is a fit parameter and

        .. math::

             Q = \begin{pmatrix} 1 & 0 & 0 \\
                                 0 & 0 & 0 \\
				 0 & 0  & 0\\
				 \end{pmatrix} 
        
        Is a tensor with $FA=1$. That is, without any radial component.

        """
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
                                      over_sample=None,  # Always None
                                      mode=mode,
                                      iso_diffusivity=iso_diffusivity,
                                      verbose=verbose)


        self.model_form = model_form
        self.iso_pred_sig = np.exp(-self.bvals[self.b_idx][0] * iso_diffusivity)

        # Over-ride the setting of the params file name in the super-class, so
        # that we can add the model form into the file name (and run on all
        # model-forms for the same data...):

        # Introspect to figure out what name the current class has:
        this_class = str(self.__class__).split("'")[-2].split('.')[-1]

        # Go on and set it: 
        self.params_file = params_file_resolver(self,
                                                this_class + model_form,
                                                params_file=params_file)


        # Choose the prediction function based on the model form:
        if self.model_form == 'constrained':
            self.pred_func = self._pred_sig_constrained
        elif self.model_form=='flexible':
            self.pred_func = self._pred_sig_flexible
        elif self.model_form == 'ball_and_stick':
            self.pred_func = self._pred_sig_ball_and_stick

        if self.mode == 'relative_signal':
            self.fit_signal = self._flat_relative_signal
        elif self.mode == 'signal_attenuation':
            self.fit_signal = 1-self._flat_relative_signal
        elif self.mode == 'normalize':
            e_s = "Mode normalize doesn't make sense in CanonicalTensorModelOpt"
            raise ValueError(e_s)
        else:
            e_s = "Mode %s not recognized"
            raise ValueError(e_s)            


    @desc.auto_attr
    def model_params(self):
        """
        Find the model parameters using least-squares optimization.
        """
        if self.model_form == 'constrained':
            n_params = 3
        elif self.model_form=='flexible' or self.model_form == 'ball_and_stick':
            n_params = 4 
        else:
            e_s = "%s is not a recognized model form"% self.model_form
            raise ValueError(e_s)

        params = np.empty((self.fit_signal.shape[0],n_params))

        if self.verbose:
            print('Fitting CanonicalTensorModelOpt:')
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        # Initialize the starting conditions for the first voxel
        if self.model_form == 'constrained':
            this_params = 0, 0, np.mean(self.fit_signal[0])
        elif (self.model_form=='flexible' or
              self.model_form=='ball_and_stick'):
            this_params = (0, 0, np.mean(self.fit_signal[0]),
                           np.mean(self.fit_signal[0]))

        for vox in xrange(self.fit_signal.shape[0]):
            # From the second voxel and onwards, we use the end point of the
            # last voxel as the starting point for this voxel:
            start_params = this_params

            # Do the least-squares fitting (setting tolerance to a rather
            # lenient value?):
            this_params, status = opt.leastsq(self._err_func,
                                              start_params,
                                    args=(self.fit_signal[vox]),
                                              ftol=10e-5
                                              )
            params[vox] = this_params

            if self.verbose: 
                prog_bar.animate(vox, f_name=f_name)

        out_params = ozu.nans(self.signal.shape[:3] + (n_params,))
        out_params[self.mask] = np.array(params).squeeze()

        return out_params

    @desc.auto_attr
    def fit(self):
        """
        Predict the signal from CanonicalTensorModelOpt
        """
        if self.verbose:
            s = "Predicting signal from CanonicalTensorModelOpt. "
            s += "Fit to %s model"%self.model_form
            print(s)

        out_flat = np.empty(self._flat_signal.shape)
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            this_pred = self.pred_func(flat_params[vox])
            if self.mode == 'signal_attenuation':
                this_pred = 1 - this_pred
            out_flat[vox] = this_pred * self._flat_S0[vox]
        
        out = ozu.nans(self.signal.shape)
        out[self.mask] = out_flat

        return out

    def _tensor_helper(self, theta, phi):
            """
            This code is used in all three error functions, so we write it out
            only once here
            """
            # Convert to cartesian coordinates:
            x,y,z = geo.sphere2cart(1, theta, phi)
            bvec = [x,y,z]
            # decompose is an auto-attr of the tensor object, so is only run once
            # and then cached:x
            evals, evecs = self.response_function.decompose
            rot = ozu.calculate_rotation(bvec, evecs[0])
            rot_evecs = evecs * rot
            rot_tensor = ozt.tensor_from_eigs(rot_evecs,
                                              evals,
                                              self.bvecs[:,self.b_idx],
                                              self.bvals[:,self.b_idx])
            return rot_tensor


    def _check_constraints(self, constraints):
        """
        Helper to check for optimization bounding constraints
        """
        # If any one of the constraints is violated:
        for var, lb, ub in constraints:
            if var<lb or var>ub:
                return True

        # Otherwise:
        return False
         

    def _pred_sig_flexible(self, params, check_constraints=True):
            """
            This is the signal prediction for the fully flexible model. 
            """
            theta, phi, tensor_w, iso_w = params
            if check_constraints:
                if self._check_constraints([[theta, 0, np.pi],
                                            [phi, -np.pi, np.pi],
                                            [tensor_w, 0, np.inf], # Weights are
                                                                   # non-negative
                                            [iso_w, 0, np.inf]]):
                    return np.inf
                
            rot_tensor = self._tensor_helper(theta, phi)
            # Relative to an S0=1:
            return (tensor_w * rot_tensor.predicted_signal(1) +
                    iso_w * self.iso_pred_sig)


    def _pred_sig_constrained(self, params, check_constraints=False):
            """
            This is the signal prediction for the constrained model. 
            """
            theta, phi, w = params

            if check_constraints:
                if self._check_constraints([[theta, 0, np.pi],
                                       [phi, -np.pi, np.pi],
                                       [w, 0, 1]]): # Weights are 0-1
                    return np.inf


            rot_tensor = self._tensor_helper(theta, phi)
            # Relative to an S0=1:
            return (1-w) * self.iso_pred_sig + w * rot_tensor.predicted_signal(1)


    def _pred_sig_ball_and_stick(self, params, check_constraints=False):
            """
            This is the signal prediction for the ball-and-stick model
            """
            theta, phi, w, d = params

            if check_constraints:
                if self._check_constraints([[theta, 0, np.pi],
                                            [phi, -np.pi, np.pi],
                                            [w, 0, 1], # Weights are 0-1
                                            [d, 0, np.inf]]): # Diffusivity is
                                                         # non-negative 
                    return np.inf
            
            # In each one we have a different axial diffusivity for the response
            # function. Simply multiply it by the current d:
            # Need to replace the canonical tensor with a 
            self.response_function =  ozt.Tensor(np.diag([1, 0, 0]),
                              self.bvecs[:, self.b_idx], self.bvals[self.b_idx])

            self.response_function.Q = self.response_function.Q * d
            rot_tensor = self._tensor_helper(theta, phi)
            return (1-w) * d + w * rot_tensor.predicted_signal(1)

        
    def _err_func(self, params, vox_sig):
            """
            This is the error function for the 'ball and stick' model. 
            """
            # During fitting, you want to check the fitting bounds:
            pred_sig = self.pred_func(params, check_constraints=True)
            return pred_sig - vox_sig

    
    def diffusion_distance(self, vertices=None):
        """
        Calculate the diffusion distance on a novel set of vertices

        Parameters
        ----------
        vertices : an n by 3 array

        """
        # If none are provided, use the measurement points:
        if vertices is None:
            self.bvecs[:, self.b_idx]

        # Start by generating the values of the rotations we use in these
        # coordinates on the sphere 
        rotations = self._calc_rotations(vertices, mode='distance')
        
        if self.verbose:
            print("Predicting signal from CanonicalTensorModel")
            
        out_flat = np.empty((self._flat_signal.shape[0], vertices.shape[-1]))
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_params[vox, 1]):
                this_dist = flat_params[vox,1] * rotations[flat_params[vox,0]]
                out_flat[vox]= this_dist
            else:
                out_flat[vox] = np.nan
                
        out = ozu.nans(self.signal.shape[:3] + (vertices.shape[-1], ))
        out[self.mask] = out_flat

        return out
