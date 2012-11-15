"""

Analyze the CSD model

"""

import inspect

import numpy as np
from scipy.special import sph_harm

import nibabel as ni
import dipy.core.geometry as geo
import dipy.reconst.recspeed as recspeed
import dipy.core.sphere as sphere

import osmosis.tensor as ozt
import osmosis.utils as ozu
import osmosis.descriptors as desc

from osmosis.model.base import BaseModel, SCALE_FACTOR
from osmosis.model.canonical_tensor import AD,RD

class _SphericalHarmonicResponseFunction(object):
    """
    This is a helper class for the SphericalHarmonicsModel.
    If you don't provide a file with the coefficients to the spherical
    harmonics basis set, it uses AD and RD and the model will use  a Tensor.

    Otherwise, it uses its own idiosyncratic set of stuff, implemented here:
    """
    def __init__(self, SHM):
        """
        Initialization for the helper class. Should get an already initialized
        SphericalHarmonicsModel as input and uses the information from it as
        the basis for doing what it does.
        """
        if SHM.verbose:
            print("Using SH-based response function")
        self.coeffs = np.loadtxt(SHM.response_file)
        self.n_coeffs = len(self.coeffs)
        self.bvecs = SHM.bvecs[:, SHM.b_idx]
        
    @desc.auto_attr
    def rotations(self):
        """
        Calculate the response function for alignment with each one of the
        b vectors
        """
        out = []
        for idx, bvec in enumerate(self.bvecs.T):
            rot = ozu.calculate_rotation(bvec, [1,0,0])
            bvecs = np.asarray(np.dot(rot, self.bvecs)).squeeze()
            r, theta, phi = geo.cart2sphere(bvecs[0],bvecs[1],bvecs[2])

            sph_harm_set = []
            degree = 0        
            for order in np.arange(0, 2 * self.n_coeffs, 2):
                sph_harm_set.append(np.real(sph_harm(degree, order, theta, phi)))

            sph_harm_set = np.array(sph_harm_set)
            out.append(np.dot(self.coeffs, sph_harm_set))

        return np.array(out)

    def convolve_odf(self, odf, S0):
        """
        Calculate the convolution of the odf with the response function
        (rotated to each b vector).
        """
        return np.dot(odf, self.rotations)
        
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
                 params_file=None,
                 axial_diffusivity=None,
                 radial_diffusivity=None,
                 response_file=None,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 verbose=True):
        """
        Initialize a SphericalHarmonicsModel class instance.
        
        Parameters
        ----------
        DWI: osmosis.dwi.DWI class instance.

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
                           sub_sample=sub_sample,
                           params_file=params_file,
                           verbose=verbose) 

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

        if (axial_diffusivity is None and radial_diffusivity is None and
            response_file is None ) :
             self.ad = AD
             self.rd = RD

        elif (axial_diffusivity is not None and radial_diffusivity is not None
            and response_file is not None):
            e_s = "Need to provide information to generate canonical tensor"
            e_s += " *or* path to response file for response function. "
            e_s += "Not both!"
            raise ValueError(e_s)
        
        self.response_file = response_file

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
          diffusion profile. First, the fact the relative signal profile is
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
        r, theta, phi = geo.cart2sphere(self.bvecs[0, self.b_idx],
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
                    b[i,:] = np.imag(sph_harm(-1 * degree, order, phi, theta));
                else:
                    b[i,:] = np.real(sph_harm(degree, order, phi, theta));
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
        out = np.empty(self.signal.shape)
        # multiply these two matrices together for the estimated odf:  
        out[self.mask] = np.dot(self.model_coeffs[self.mask], self.sph_harm_set)

        return out 

    @desc.auto_attr
    def odf_peaks(self):
        """
        Calculate the value of each of the peaks in the ODF using the dipy
        peak-finding algorithm
        """
        faces = sphere.Sphere(xyz=self.bvecs[:,self.b_idx].T).faces
        odf_flat = self.odf[self.mask]
        out_flat = np.zeros(odf_flat.shape)
        for vox in xrange(odf_flat.shape[0]):
            peaks, inds = recspeed.local_maxima(odf_flat[vox], faces)
            out_flat[vox][inds] = peaks 

        out = np.zeros(self.odf.shape)
        out[self.mask] = out_flat
        return out

    @desc.auto_attr
    def crossing_index(self):
        """
        Calculate an index of crossing in each voxel. This index is an analogue
        of FA, in that it is a normalized standard deviation between the values
        of the magnitudes of the peaks, which is then normalized by the
        standard deviation of the case in which there is only 1 peak with the
        value '1'.
        """
        # Flatten and sort (small => large) 
        flat_peaks = self.odf_peaks[self.mask]
        cross_flat = np.empty(flat_peaks.shape[0])
        for vox in xrange(cross_flat.shape[0]):
            # Normalize all the peaks by the 2-norm of the vector:
            peaks_norm = flat_peaks[vox]/np.sqrt(np.dot(flat_peaks[vox],
                                                        flat_peaks[vox]))
            non_zero_idx = np.where(peaks_norm>0)[0]

            # Deal with some corner cases - if there is no peak, we define this
            # to be 0: 
            if len(non_zero_idx) == 0:
                cross_flat[vox] = 0
            # If there's only one peak, we define it to be 1: 
            elif len(non_zero_idx) == 1:
                cross_flat[vox] = 1
            # Otherwise, we need to do some math: 
            else: 
                std_peaks = (np.std(peaks_norm[non_zero_idx]))
                std_norm = np.std(np.hstack([1, np.zeros(len(non_zero_idx)-1)]))
                cross_flat[vox] = std_peaks/std_norm
            
        cross = ozu.nans(self.data.shape[:3])
        cross[self.mask] = cross_flat
        return cross
        

    @desc.auto_attr
    def response_function(self):
        """
        A canonical tensor that describes the presumed response of a single
        fiber 
        """
        if self.response_file is None:
            if self.verbose:
                print("Using tensor-based response function")
            return ozt.Tensor(np.diag([self.ad, self.rd, self.rd]),
                          self.bvecs[:,self.b_idx],
                          self.bvals[self.b_idx])

        else:
            return _SphericalHarmonicResponseFunction(self)
        
    @desc.auto_attr
    def fit(self):
        """
        This is the signal estimated from the odf.
        """
        if self.verbose:
            print("Predicting signal from SphericalHarmonicsModel")
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        # Reshape the odf to be one voxel per row:
        flat_odf = self.odf[self.mask]
        pred_sig = np.empty(flat_odf.shape)

            
        for vox in xrange(pred_sig.shape[0]):
            # Predict based on the convolution:
            this_pred_sig = self.response_function.convolve_odf(
                                                    flat_odf[vox],
                                                    self._flat_S0[vox])

            # We might have a scaling and an offset in addition, so let's fit
            # those in each voxel based on the signal:
            a,b = np.polyfit(this_pred_sig, self._flat_signal[vox], 1)
            pred_sig[vox] = a*this_pred_sig + b
            
            if self.verbose:
                prog_bar.animate(vox, f_name=f_name)

        # Pack it back into a volume shaped thing: 
        out = ozu.nans(self.signal.shape)
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

    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        The principal direction of diffusion as determined by the maximum of
        the estimated ODF in each voxel.
        """
        
        flat_odf = self.odf[self.mask]
        out_flat = np.empty((flat_odf.shape[0], 3))
        for vox in xrange(out_flat.shape[0]):
            this_odf = flat_odf[vox]
            
            out_flat[vox] =\
                self.bvecs[:, self.b_idx].T[np.argmax(this_odf)]

        out = ozu.nans(self.shape[:3] + (3,))
        out[self.mask] = out_flat
        return out
