"""

This module is used to construct and solve models of diffusion data 

"""
import os
import warnings

import numpy as np

# We want to try importing numexpr for some array computations, but we can do
# without:
try:
    import numexpr
    has_numexpr = True
except ImportError: 
    has_numexpr = False
    
# Import stuff for sparse matrices:
import scipy.sparse as sps

import scipy.linalg as la
import scipy.stats as stats
from scipy.special import sph_harm

import dipy.reconst.dti as dti
import dipy.core.geometry as geo
import nibabel as ni

import microtrack.descriptors as desc
import microtrack.fibers as mtf
import microtrack.tensor as mtt
import microtrack.utils as mtu
import microtrack.boot as boot


# Global constants for this module:
AD = 1.5
RD = 0.5
# This converts b values from , so that it matches the units of ADC we use in
# the Stejskal/Tanner equation: 
SCALE_FACTOR = 1000

class DWI(desc.ResetMixin):
    """
    A class for representing dwi data
    """
            
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):
        """
        Initialize a DWI object

        Parameters
        ----------
        data: str or array
            The diffusion weighted mr data provided either as a full path to a
            nifti file containing the data, or as a 4-d array.

        bvecs: str or array
            The unit vectors describing the directions of data
            acquisition. Either an 3 by n array, or a full path to a text file
            containing the 3 by n data.

        bvals: str or array
            The values of b weighting in the data acquisition. Either a 1 by n
            array, or a full path to a text file containing the values.

        affine: optional, 4 by 4 array
            The affine provided in the file can be overridden by explicitely
            setting this input variable. If this is left as None, one of two
            things will happen. If the 'data' input was a file-name, the affine
            will be read from that file. Otherwise, a warning will be issued
            and affine will default to np.eye(4).

        mask: optional, 3-d array
            When provided, used as a boolean mask into the data for access. 

        sub_sample: int or array of ints.
           If we want to sub-sample the DWI data on the sphere (in the bvecs),
           we can do one of two things: 

        1. If sub_sample is an integer, that number of random bvecs will be
           chosen from the data.

        2. If an array of indices is provided, these will serve as indices into
        the last dimension of the data and only that part of the data will be
        used

        """
        
        # All inputs are handled essentially the same. Inputs can be either
        # strings, in which case file reads are required, or arrays, in which
        # case no file reads are needed and we assign these arrays into the
        # attributes:
        for name, val in zip(['data', 'bvecs', 'bvals'],
                             [data, bvecs, bvals]): 
            if isinstance(val, str):
                exec("self.%s_file = '%s'"% (name, val))
            elif isinstance(val, np.ndarray):
                # This time we need to give it the name-space:
                exec("self.%s = val"% name, dict(self=self, val=val))
            else:
                e_s = "%s seems to be neither an array, "% name
                e_s += "nor a file-name\n"
                e_s += "The value provided was: %s" % val
                raise ValueError(e_s)

        # You might have to scale the bvalues by some factor, so that the units
        # come out correctly in the adc calculation:
        self.bvals /= scaling_factor
        
        # You can provide your own affine, if you want and that bypasses the
        # class method provided below as an auto-attr:
        if affine is not None:
            self.affine = np.matrix(affine)

        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # If it's a string, assume it's the full-path to a nifti file with
            # a binary mask: 
            if isinstance(mask, str):
                mask = ni.load(mask).get_data()
            self.mask = np.array(mask, dtype=bool)

        else:
            # Spatial mask (take only the spatial dimensions):
            self.mask = np.ones(self.data.shape[:3], dtype=bool)

        if sub_sample is not None:
            if np.iterable(sub_sample):
                idx = sub_sample
            else:
                idx = boot.subsample(self.bvecs[:,self.b_idx].T, sub_sample)[1]

            self.b_idx = self.b_idx[idx]
            # At this point, signal will be taken according to these
            # sub-sampled indices:
            self.data = np.concatenate([self.signal,
                                        self.data[:,:,:,self.b0_idx]],-1)

            self.b0_idx = np.arange(len(self.b0_idx))

            self.bvecs = np.concatenate([np.zeros((3,len(self.b0_idx))),
                                        self.bvecs[:, self.b_idx]],-1)

            self.bvals = np.concatenate([np.zeros(len(self.b0_idx)),
                                         self.bvals[self.b_idx]])
            self.b_idx = np.arange(len(self.b0_idx), len(self.b0_idx) + len(idx))


    @desc.auto_attr
    def shape(self):
        """
        Get the shape of the data. If possible, don't even load it from file to
        get that. 
        """

        # It must have been in an array
        if not hasattr(self, 'data_file'):
            # No reason not to refer to it directly:
            return self.data.shape
        
        # The data is in a file, and you might not have loaded it yet:
        else:
            # No need to actually load it yet:
            return ni.load(self.data_file).get_shape()
            
    @desc.auto_attr
    def bvals(self):
        """
        If bvals were not provided as an array, read them from file
        """ 
        return np.loadtxt(self.bvals_file)
        
    @desc.auto_attr
    def bvecs(self):
        """
        If bvecs were not provided as an array, read them from file
        """ 
        return np.loadtxt(self.bvecs_file)

    @desc.auto_attr
    def data(self):
        """
        Load the data from file
        """
        return ni.load(self.data_file).get_data()

    @desc.auto_attr
    def affine(self):
        """
        Get the affine transformation of the data to world coordinates
        (relative to acpc)
        """
        if hasattr(self, 'data_file'):
            # This means that there might be an affine to read in from file.
            return np.matrix(ni.load(self.data_file).get_affine())
        else:
            w_s = "DWI data generated from array. Affine will be set to"
            w_s += " np.eye(4)"
            warnings.warn(w_s)
            return np.matrix(np.eye(4))

    @desc.auto_attr
    def _flat_data(self):
        """
        Get the flat data only in the mask
        """        
        return np.reshape(self.data[self.mask],
                          (-1, self.bvecs.shape[-1]))

    @desc.auto_attr
    def _flat_S0(self):
        """
        Get the signal in the b0 scans in flattened form (only in the mask)
        """
        return np.mean(self._flat_data[:,self.b0_idx], -1)

    @desc.auto_attr
    def _flat_signal(self):
        """
        Get the signal in the diffusion-weighted volumes in flattened form
        (only in the mask).
        """
        return self._flat_data[:,self.b_idx]


    def noise_ceiling(self, DWI2, correlator=stats.pearsonr, r_idx=0):
        """
        Calculate the r-squared of the correlator function provided, in each
        voxel (across directions, including b0's (?) ) between this class
        instance and another class  instance, provided as input. r_idx points
        to the location of r within the tuple returned by the correlator callable
        """
                
        val = np.empty(self._flat_data.shape[0])

        for ii in xrange(len(val)):
            val[ii] = correlator(self._flat_data[ii],
                                 DWI2._flat_data[ii])[0] 

        if has_numexpr:
            r_squared = numexpr.evaluate('val**2')
        else:
            r_squared = val**2

        # Re-package it into a volume:
        out = np.nan*np.ones(self.data.shape[:3])
        out[self.mask] = r_squared
    
        return out 

    @desc.auto_attr
    def b_idx(self):
        """
        The indices into non-zero b values
        """
        return np.where(self.bvals > 0)[0]
        
    @desc.auto_attr
    def b0_idx(self):
        """
        The indices into zero b values
        """
        return np.where(self.bvals==0)[0]

    @desc.auto_attr
    def S0(self):
        """
        Extract and average the signal for volumes in which no b weighting was
        used (b0 scans)
        """
        return np.mean(self.data[...,self.b0_idx],-1).squeeze()
        
    @desc.auto_attr
    def signal(self):
        """
        The signal in b-weighted volumes
        """
        return self.data[...,self.b_idx].squeeze()


class BaseModel(DWI):
    """
    Base-class for models.
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):
        """
        A base-class for models based on DWI data.

        Parameters
        ----------

        scaling_factor: int, defaults to 1000.
           To get the units in the S/T equation right, how much do we need to
           scale the bvalues provided.
        
        """
        # DWI should already have everything we need: 
        DWI.__init__(self,
                         data,
                         bvecs,
                         bvals,
                         affine=affine,
                         mask=mask,
                         scaling_factor=scaling_factor,
                         sub_sample=sub_sample) 
        
    @desc.auto_attr
    def fit(self):
        """
        Each model will have a model prediction, which is always in this class
        method. This prediction is used in other methods, such as 'residuals'
        and 'r_squared', etc.

        In this particular case, we set fit to be exactly equal to the
        signal. This should make testing easy :-) 
        """
        return self.signal

    @desc.auto_attr
    def _flat_fit(self):
        """
        Extract a flattened version of the fit, defined for masked voxels
        """
        
        return self.fit[self.mask].reshape((-1, self.signal.shape[-1])) 
        

    def _correlator(self, func, r_idx):
        """
        Helper function that uses a callable "func" to apply between two 1-d
        arrays. These 1-d arrays can have different outputs and the one we
        always want is the one which is r_idx into the output tuple 
        """
        r = np.empty(self._flat_signal.shape[0])
        
        for ii in xrange(len(self._flat_signal)):
            outs = func(self._flat_signal[ii] , self._flat_fit[ii])
            r[ii] = outs[r_idx]
            
        if has_numexpr:
            r_squared = numexpr.evaluate('r**2')
        else:
            r_squared = r**2

        # Re-package it into a volume:
        out = np.nan*np.ones(self.data.shape[:3])
        out[self.mask] = r_squared
        return out         

    @desc.auto_attr
    def r_squared(self):
        """
        The r-squared ('explained variance') value in each voxel
        """
        return self._correlator(stats.pearsonr, 0)
    
    @desc.auto_attr
    def R_squared(self):
        """
        The R-squared ('coefficient of determination' from a linear model fit)
        in each voxel
        """
        return self._correlator(stats.linregress, 2)

    @desc.auto_attr
    def coeff_of_determination(self):
        """
        Explained variance as: 100 *(1-\frac{RMS(residuals)}{RMS(signal)})

        http://en.wikipedia.org/wiki/Coefficient_of_determination
        
        """
        return mtu.explained_variance(self.fit, self.signal)

    @desc.auto_attr
    def RMSE(self):
        """
        The square-root of the mean of the squared residuals
        """

        # Preallocate the output: 
        out = np.nan*np.ones(self.data.shape[:3])

        res = self.residuals[self.mask]
        
        if has_numexpr:
            out[self.mask] = np.sqrt(np.mean(
                             numexpr.evaluate('res ** 2'), -1))
        else:
            out[self.mask] = np.sqrt(np.mean(np.power(res, 2),-1))
        
        return out

    
    @desc.auto_attr
    def residuals(self):
        """
        The prediction-subtracted residual in each voxel
        """
        out = np.nan*np.ones(self.signal.shape)
        sig = self.signal[self.mask]
        fit = self.fit[self.mask]
        
        if has_numexpr:
            out[self.mask] = numexpr.evaluate('sig - fit')

        else:
            out[self.mask] = sig - fit

        return out



class TensorModel(BaseModel):

    """
    A class for representing and solving a simple forward model. Just the
    diffusion tensor.
    
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 tensor_file=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):
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


        tensor_file: A file to cache the initial tensor calculation in. If this
        file already exists, we pull the tensor fit out of it. Otherwise, we
        calculate the tensor fit and save this file with the params of the
        tensor fit. 
        
        """
        # Initialize the super-class:
        BaseModel.__init__(self,
                           data,
                           bvecs,
                           bvals,
                           affine=None,
                           mask=mask,
                           scaling_factor=scaling_factor,
                           sub_sample=sub_sample) 

        if tensor_file is not None: 
            self.tensor_file = tensor_file
        else:
            # If DWI has a file-name, construct a file-name out of that: 
            if hasattr(self, 'data_file'):
                path, f = os.path.split(self.data_file)
                # Need to deal with the double-extension in '.nii.gz':
                file_parts = f.split('.')
                name = file_parts[0]
                extension = ''
                for x in file_parts[1:]:
                    extension = extension + '.' + x
                self.tensor_file = os.path.join(path, name + 'TensorModel' +
                                              extension)
            else:
                # Otherwise give up and make a file right here with a generic
                # name: 
                self.tensor_file = 'DTI.nii.gz'
        
    @desc.auto_attr
    def model_params(self):
        """
        The diffusion tensor parameters estimated from the data, using dipy.
        If this calculation has already occurred, just load the data from a
        nifti file, which has shape x by y by z by 12, where the last dimension
        is the model params:

        evecs (9) + evals (3)
        
        """
        # The file already exists: 
        if os.path.isfile(self.tensor_file):
            return ni.load(self.tensor_file).get_data()
        else: 
            mp = dti.Tensor(self.data,
                            self.bvals,
                            self.bvecs.T,
                            self.mask).model_params 

            # Save the params for future use: 
            params_ni = ni.Nifti1Image(mp, self.affine)
            params_ni.to_filename(self.tensor_name)
            # And return the params for current use:
            return mp

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
    def fractional_anisotropy(self):
        """
        .. math::

            FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                        \lambda_3)^2+(\lambda_2-lambda_3)^2}{\lambda_1^2+
                        \lambda_2^2+\lambda_3^2} }

        """
        out = np.nan * np.ones(self.data.shape[:3])
        
        lambda_1 = self.evals[..., 0][self.mask]
        lambda_2 = self.evals[..., 1][self.mask]
        lambda_3 = self.evals[..., 2][self.mask]

        out[self.mask] = mtu.fractional_anisotropy(lambda_1, lambda_2, lambda_3)
        return out

    @desc.auto_attr
    def radial_diffusivity(self):
        return np.mean(self.evals[...,1:],-1)

    @desc.auto_attr
    def axial_diffusivity(self):
        return self.evals[...,0]

    # Self Diffusion Tensor, taken from dipy.reconst.dti:
    @desc.auto_attr
    def tensors(self):
        out = np.nan * np.ones(self.evecs.shape)
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
    def apparent_diffusion_coef(self):
        out = np.empty(self.signal.shape)
        tensors_flat = self.tensors[self.mask].reshape((-1,3,3))
        adc_flat = np.empty(self.signal[self.mask].shape)

        for ii in xrange(len(adc_flat)):
            adc_flat[ii] = mtt.apparent_diffusion_coef(
                                        self.bvecs[:,self.b_idx],
                                        tensors_flat[ii])

        out[self.mask] = adc_flat
        return out

    @desc.auto_attr
    def fit(self):
        adc_flat = self.apparent_diffusion_coef[self.mask]
        fit_flat = np.empty(adc_flat.shape)
        out = np.empty(self.signal.shape)

        for ii in xrange(len(fit_flat)):
            fit_flat[ii] = mtt.stejskal_tanner(self._flat_S0[ii],
                                               self.bvals[:, self.b_idx],
                                               adc_flat[ii])

        out[self.mask] = fit_flat
        return out


class SphericalHarmonicsModel(BaseModel):
    """
    A class for evaluating spherical harmonic models. This assumes that a CSD
    model was already fit somehow. Presumably by using mrtrix    
    """ 
    
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 model_coeffs,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):
        """
        Initialize a SphericalHarmonicsModel class instance.
        
        Parameters
        ----------
        DWI: microtrack.dwi.DWI class instance.

        model_coefficients: ndarray
           Coefficients for a SH model, organized according to the conventions
           used by mrtrix (see sph_harm_set for details).
        
        """
        # Initialize the super-class:
        BaseModel.__init__(self,
                           data,
                           bvecs,
                           bvals,
                           affine=affine,
                           mask=mask,
                           scaling_factor=scaling_factor,
                           sub_sample=sub_sample) 

        # If it's a string, assume it's a full path to a nifti file: 
        if isinstance(model_coeffs,str):
            self.model_coeffs = ni.load(model_coeffs).get_data()
        else:
            # Otherwise, it had better be an array:
            self.model_coeffs = model_coeffs

        self.L = self._calculate_L(self.model_coeffs.shape[-1])
        self.n_params = self.model_coeffs.shape[-1]

        self.ad = axial_diffusivity
        self.rd = radial_diffusivity
        
    @desc.auto_attr
    def sph_harm_set(self):
        """
        Calculate the spherical harmonics, provided n parameters (corresponding
        to nc = (L+1) * (L+2)/2 with L being the maximal harmonic degree for
        the set of bvecs of the object

        Note
        ----

        1. This was written according to the documentation of mrtrix's
        'csdeconv'. The following is taken from there:  

          Note that this program makes use of implied symmetries in the
          diffusion profile. First, the fact the signal attenuation profile is
          real implies that it has conjugate symmetry, i.e. Y(l,-m) = Y(l,m)*
          (where * denotes the complex conjugate). Second, the diffusion
          profile should be antipodally symmetric (i.e. S(x) = S(-x)), implying
          that all odd l components should be zero. Therefore, this program
          only computes the even elements.

          Note that the spherical harmonics equations used here differ slightly
          from those conventionally used, in that the (-1)^m factor has been
          omitted. This should be taken into account in all subsequent
          calculations.

          Each volume in the output image corresponds to a different spherical
          harmonic component, according to the following convention: [0]    
          Y(0,0)  [1] Im {Y(2,2)} [2] Im {Y(2,1)} [3]     Y(2,0) [4] Re
          {Y(2,1)} [5] Re {Y(2,2)}  [6] Im {Y(4,4)} [7] Im {Y(4,3)} etc... 

          
        2. Take heed that it seems that scipy's sph_harm actually has the
        order/degree in reverse order than the convention used by mrtrix, so
        that needs to be taken into account in the calculation below

        """
                
        # Convert to spherical coordinates:
        r,theta,phi = geo.cart2sphere(self.bvecs[0, self.b_idx],
                                      self.bvecs[1, self.b_idx],
                                      self.bvecs[2, self.b_idx])

        # Preallocate:
        b = np.empty((self.model_coeffs.shape[-1], theta.shape[0]))
    
        i = 0;
        # Only even order are taken:
        for order in np.arange(0, self.L + 1, 2): # Go to L, inclusive!
           for degree in np.arange(-order,order+1):
                # In negative degrees, take the imaginary part: 
                if degree < 0:  
                    b[i,:] = np.imag(sph_harm(-1 * degree, order, theta, phi));
                else:
                    b[i,:] = np.real(sph_harm(degree, order, theta, phi));
                i = i+1;
        return b

    @desc.auto_attr
    def odf(self): 
        """        
        Generate a volume with dimensions (x,y,z, n_bvecs) where each voxel has:

        .. math::

          \sum{w_i, b_i}

        Where $b_i$ are the basis set functions defined from the spherical
        harmonics and $w_i$ are the model coefficients estimated with CSD.

        This a unit-less estimate of the orientation distribution function,
        based on the estimation of the SH coefficients. This needs to be
        convolved with a "response function", a canonical tensor, to calculate
        back the estimated signal. 
        """
        volshape = self.model_coeffs.shape[:3] # Disregarding the params
                                               # dimension
        n_vox = np.sum(self.mask) # These are the voxels we'll look at
        n_weights = self.model_coeffs.shape[3]  # This is the params dimension 

        # Reshape it so that we can multiply for all voxels in one fell swoop:
        d = np.reshape(self.model_coeffs[self.mask], (n_vox, n_weights))

        out = np.empty(self.signal.shape)
        
        # multiply these two matrices together for the estimated odf:  
        out[self.mask] = np.asarray(np.matrix(d) *
                                     np.matrix(self.sph_harm_set))

        return out 
    

    @desc.auto_attr
    def response_function(self):
        """
        A canonical tensor that describes the presumed response of a single
        fiber 
        """
        return mtt.Tensor(np.diag([self.ad, self.rd, self.rd]),
                          self.bvecs[:,self.b_idx],
                          self.bvals[self.b_idx])
        

    @desc.auto_attr
    def fit(self):
        """
        This is the signal estimated from the odf.
        """        
        # Reshape the odf to be one voxel per row:
        flat_odf = self.odf[self.mask]
        pred_sig = np.empty(flat_odf.shape)

        for vox in range(pred_sig.shape[0]):
            pred_sig[vox] = self.response_function.convolve_odf(
                                                    flat_odf[vox],
                                                    self._flat_S0[vox])

        # Pack it back into a volume shaped thing: 
        out = np.ones(self.signal.shape) * np.nan
        out[self.mask] = pred_sig  
        return out
        
        
    def _calculate_L(self,n):
        """
        Calculate the maximal harmonic order (L), given that you know the
        number of parameters that were estimated. This proceeds according to
        the following logic:

        .. math:: 

           n = \frac{1}{2} (L+1) (L+2)

           \rarrow 2n = L^2 + 3L + 2
           \rarrow L^2 + 3L + 2 - 2n = 0
           \rarrow L^2 + 3L + 2(1-n) = 0

           \rarrow L_{1,2} = \frac{-3 \pm \sqrt{9 - 8 (1-n)}}{2}

           \rarrow L{1,2} = \frac{-3 \pm \sqrt{1 + 8n}}{2}


        Finally, the positive value is chosen between the two options. 
        """

        L1 = (-3 + np.sqrt(1+ 8 *n))/2
        L2 = (-3 - np.sqrt(1+ 8 *n))/2

        return max([L1,L2])

    
class FiberModel(BaseModel):
    """
    
    A class for representing and solving predictive models based on
    tractography solutions.
    
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 FG,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):
        """
        Parameters
        ----------
        DWI: A microtrack.dwi.DWI object, or a list containing: [the name of
             nifti file, from which data should be read, bvecs file, bvals file]
        
        FG: a microtrack.fibers.FiberGroup object, or the name of a pdb file
            containing the fibers to be read in using mtf.fg_from_pdb

        axial_diffusivity: The axial diffusivity of a single fiber population.

        radial_diffusivity: The radial diffusivity of a single fiber population.

        scaling_factor: This scales the b value for the Stejskal/Tanner equation
        
        """
        # Initialize the super-class:
        BaseModel.__init__(self,
                            data,
                            bvecs,
                            bvals,
                            affine=None,
                            mask=None,
                            scaling_factor=scaling_factor,
                            sub_sample=None)

        self.axial_diffusivity = axial_diffusivity
        self.radial_diffusivity = radial_diffusivity

        # The only additional thing is that this one also has a fiber group,
        # which is xformed to match the coordinates of the DWI:
        self.FG = FG.xform(self.affine.getI(), inplace=False)


        # XXX There's got to be a way to get a mask here, which will refer only
        # to where the fibers are. 
        
    @desc.auto_attr
    def fg_idx(self):
        """
        Indices into the coordinates of the fiber-group
        """
        return self.fg_coords.astype(int)
    
    @desc.auto_attr
    def fg_coords(self):
        """
        All the coords of all the fibers  
        """
        return self.FG.coords

    @desc.auto_attr
    def fg_idx_unique(self):
        return mtu.unique_rows(self.fg_idx)

    
    @desc.auto_attr
    def matrix(self):
        """
        The matrix of fiber-contributions to the DWI signal.
        """
        # Assign some local variables, for shorthand:
        vox_coords = self.fg_idx_unique
        n_vox = vox_coords.shape[-1]
        n_bvecs = self.b_idx.shape[0]
        n_fibers = self.FG.n_fibers
        matrix_dims = np.array([n_vox * n_bvecs, n_fibers])
        matrix_len = matrix_dims.prod()

        # Preallocate these:
        matrix_sig = np.zeros(matrix_len)
        matrix_row = np.zeros(matrix_len)
        matrix_col = np.zeros(matrix_len)
        
        for f_idx, fiber in enumerate(self.FG.fibers):
            start = f_idx * n_vox * n_bvecs
            # These are easy:
            matrix_row[start:start+matrix_dims[0]] = (np.arange(matrix_dims[0]))
            matrix_col[start:start+matrix_dims[0]] = f_idx * np.ones(n_vox *
                                                                     n_bvecs)
            # Here comes the tricky part:
            print "working on fiber %s"%(f_idx + 1)
            fiber_idx =  fiber.coords.astype(int)
            fiber_pred = fiber.predicted_signal(
                                self.bvecs[:, self.b_idx],
                                self.bvals[:, self.b_idx],
                                self.axial_diffusivity,
                                self.radial_diffusivity,
                                self.S0[fiber_idx[0],
                                        fiber_idx[1],
                                        fiber_idx[2]]
                                ).ravel()
            # Do we really have to do this one-by-one?
            for i in xrange(fiber_idx.shape[-1]):
                arr_list = [np.where(self.fg_idx_unique[j]==fiber_idx[:,i][j])[0]
                            for j in [0,1,2]]
                this_idx = mtu.intersect(arr_list)
                # Sum the signals from all the fibers/all the nodes in each
                # voxel, by summing over the predicted signal from each fiber
                # through all its
                # nodes and just adding it in:
                for k in this_idx:
                    matrix_sig[start + k * n_bvecs:
                               start + k * n_bvecs + n_bvecs] += \
                        fiber_pred[i:i+n_bvecs]

        #Put it all in one sparse matrix:
        return sps.coo_matrix((matrix_sig,[matrix_row, matrix_col]))
    
    @desc.auto_attr
    def flat_signal(self):
        """
        XXX - need to change the name of this. This refers to something else
        usually
        
        The signal in the voxels corresponding to where the fibers pass through.
        """ 
        return self.signal[self.fg_idx_unique[0],
                               self.fg_idx_unique[1],
                               self.fg_idx_unique[2]].ravel()

