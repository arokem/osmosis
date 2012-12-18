import inspect

import numpy as np
import scipy.optimize as opt

import nibabel as ni
import dipy.core.geometry as geo

import osmosis.tensor as ozt
import osmosis.utils as ozu
import osmosis.descriptors as desc
from osmosis.model.canonical_tensor import CanonicalTensorModel
from osmosis.model.base import SCALE_FACTOR


class CalibratedCanonicalTensorModel(CanonicalTensorModel):
    """
    This is another extension of the CanonicalTensorModel, which extends the
    interpertation of the different weights, by calibrating the weights to a
    particular ROI.

    Classically, we will use Corpus Callosum, or some part of it as our
    'calibration target'. In CC, we assume that the axial diffusivity of the
    canonical tensor used is the same as the diffusivity (uniform in all
    directions) of the cellular component in that part of the brain. This
    assumption is based on the idea that diffusion along the axis of the axon
    is hindered by the same kind of things that hinder diffusion inside cells:
    membranes of sub-cellular organelles, macro-molecules, etc. 

    Making this assumption we can write our non-linear model for this part of
    the brain as: 

    .. math ::

    \frac{S}{S_0} = \beta e^{-b \lambda_1} + (1-\beta)e^{-b \vec{b}Q\vect{b}^t}

    Where:

    .. math :: 

    $Q = \begin{pmatrix} \lambda_1 & 0 & 0 \\ 0 &\lambda_2 & 0 \\ 0 & 0 &
\lambda_2 \end{pmatrix}$

    is the quadratic form of the canonical tensor. Once we fit \lambda_1,
    \lambda_2 and \beta to the data from the 'calibration target', we 
    can apply these \lambda_i everywhere.

    To do that, we also need to fit the direction of the canonical tensor in
    that location, which adds two parameters to the fit. Importantly, if we
    choose a part of the brain where the direction of the principal diffusion
    direction is known (such as CC), we can reduce the optimization
    substantially, by starting things off with the canonical tensor oriented in
    the L/R direction. 
    
    """

    
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 calibration_roi,
                 params_file=None,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 verbose=True):
        """
        Initialize a CalibratedCanonicalTensorModel instance.

        Parameters
        ----------

        calibration_roi: full path to a nifti file containing zeros everywhere,
        except ones where the calibration ROI is defined. Should be already
        registered and xformed to the DWI data resolution/alignment. 

        """
        # Initialize the super-class, we set AD and RD to None, to prevent
        # things from going forward before calibration has occurred. This will
        # probably cause an error to be thrown, if calibration doesn't
        # happen. We might want to catch that error and explain it to the
        # user... 
        CanonicalTensorModel.__init__(self,
                                      data,
                                      bvecs,
                                      bvals,
                                      params_file=params_file,
                                      axial_diffusivity=None,
                                      radial_diffusivity=None,
                                      affine=affine,
                                      mask=mask,
                                      scaling_factor=scaling_factor,
                                      sub_sample=sub_sample,
                                      over_sample=over_sample,
                                      verbose=verbose)


        # This is used to initialize the optimization in each voxel.
        # The orientation parameters are chosen to be close to horizontal.
        
        self.start_params = np.pi/2, 0, 0.5, 1.5, 0
                           #theta, phi, beta, lambda1, lambda2
        self.calibration_roi = calibration_roi
        
    def _err_func(self, params, args):
        """
        Error function for the non-linear optimization 
        """

        # The fit parameters: 
        theta, phi, beta, lambda1, lambda2 = params
        # Additional argument
        vox_sig = args

        # Constraints to stabilize the fit 
        # Angles are 0=<theta<=pi 
        if theta>np.pi or theta<0:
            return np.inf
        # ... and -pi<=phi<= pi:
        if phi>np.pi or phi<-np.pi:
            return np.inf
        # No negative diffusivities: 
        if lambda1<0 or lambda2<0:
             return np.inf
        # The axial diffusivity needs to be larger than the radial diffusivity
        if lambda2 > lambda1:
            return np.inf
        # Weights between 0 and 1:
        if beta>1 or beta<0:
             return np.inf

        # Predict the signal based on the current parameter setting
        this_pred = self._pred_sig(theta, phi, beta, lambda1, lambda2)

        # The predicted signal needs to be between 0 and 1 (relative signal!):
        if np.any(this_pred>1) or np.any(this_pred<0):
            return np.inf

        # Finally, if everything is alright, return the error (leastsq will take
        # care of squaring and summing it for you):
        return (this_pred - vox_sig)

    def _pred_sig(self, theta, phi, beta, lambda1, lambda2):
        """
        The predicted signal for a particular setting of the parameters
        """

        Q = np.array([[lambda1, 0, 0],
                      [0, lambda2, 0],
                      [0, 0, lambda2]])

        # If for some reason this is not symmetrical, then something is wrong
        # (in all likelihood, this means that the optimization process is
        # trying out some crazy value, such as nan). In that case, abort and
        # return a nan:
        if not np.allclose(Q.T, Q):
            return np.nan
        
        response_function = ozt.Tensor(Q,
                                        self.bvecs[:,self.b_idx],
                                        self.bvals[:,self.b_idx])
                                        
        # Convert theta and phi to cartesian coordinates:
        x,y,z = geo.sphere2cart(1, theta, phi)
        bvec = [x,y,z]
        evals, evecs = response_function.decompose

        rot_tensor = ozt.tensor_from_eigs(
            evecs * ozu.calculate_rotation(bvec, evecs[0]),
            evals, self.bvecs[:,self.b_idx], self.bvals[:,self.b_idx])

        iso_sig = np.exp(-self.bvals[self.b_idx][0] * lambda1)
        tensor_sig =  rot_tensor.predicted_signal(1)

        return beta * iso_sig + (1-beta) * tensor_sig
        

    @desc.auto_attr
    def calibration_signal(self):
        """
        The relative signal, extracted from the calibration target ROI and
        flattened (n_voxels by n_directions)

        """
        # Need to get it from file: 
        if isinstance(self.calibration_roi, str):
            roi_mask = ni.load(self.calibration_roi).get_data()
            idx = np.where(roi_mask)
        elif isinstance(self.calibration_roi, tuple):
            idx = self.calibration_roi
        elif isinstance(self.calibration_roi, np.ndarray):
            roi_mask = self.calibration_roi
            idx = np.where(roi_mask)
            
        return np.reshape(self.relative_signal[idx],
                          (-1, self.b_idx.shape[0]))            
        
    @desc.auto_attr
    def calibrate(self):

        """"
        This is the function to perform the calibration optimization on. When
        this is done, self.AD and self.RD will be set and parameter estimation
        can proceed as in the super-class

        """

        out = np.empty((self.calibration_signal.shape[0],
                        len(self.start_params)))
        
        if self.verbose:
            print('Calibrating for AD/RD')
            prog_bar = ozu.ProgressBar(self.calibration_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        for vox in xrange(self.calibration_signal.shape[0]):
            # Perform the fitting itself:
            #out[vox], ier = leastsqbound(self._err_func,
            #                             self.start_params,
            #                             bounds = bounds,
            #                             args=(self.calibration_signal[vox]),
            #                             **optim_kwds)

            out[vox], ier = opt.leastsq(self._err_func,
                                        self.start_params,
                                        args=(self.calibration_signal[vox]))
            
            if self.verbose: 
                prog_bar.animate(vox, f_name=f_name)

        # Set the object's AD/RD according to the calibration:
        self.ad = np.median(out[:, -2])
        self.rd = np.median(out[:, -1])
        # The isotropic component diffusivity is set to be the same as the
        # axial diffusivity in the fiber component: 
        self.iso_diffusivity = self.ad

        return out


    @desc.auto_attr
    def calibration_fit(self):
        """
        Check how well the calibration model fits the signal in the calibration
        target
        """

        out = np.empty((self.calibration_signal.shape[0],
                        self.relative_signal.shape[-1]))

        # Get the calibration parameters: 
        theta, phi, beta, lambda1, lambda2 = self.calibrate.T

        for vox in xrange(out.shape[0]):
            out[vox] = self._pred_sig(theta[vox],
                                      phi[vox],
                                      beta[vox],
                                      lambda1[vox],
                                      lambda2[vox])
        return out        
