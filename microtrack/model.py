"""

This module is used to construct and solve models of diffusion data 

"""
import os

import numpy as np

# We want to try importing numexpr for some array computations, but we can do
# without:
try:
    import numexpr
    has_ne = True
except ImportError: 
    has_ne = False
    
# Import stuff for sparse matrices:
import scipy.sparse as sps
import dipy.reconst.dti as dti
import nibabel as ni

import microtrack.descriptors as desc
import microtrack.fibers as mtf
import microtrack.tensor as mtt
import microtrack.dwi as dwi
import microtrack.utils as mtu
import microtrack.boot as boot


# Global constants:
AD = 1.5
RD = 0.5
# This converts b values from , so that it matches the units of ADC we use in
# the Stejskal/Tanner equation: 
SCALE_FACTOR = 1000

class BaseModel(desc.ResetMixin):
    """
    Base-class for models.
    """
    def __init__(self,DWI,scaling_factor=SCALE_FACTOR,
                                      sub_sample=None):
        """
        A base-class for models based on DWI data.

        Parameters
        ----------
        DWI: A microtrack.dwi.DWI class instance

        scaling_factor: int, defaults to 1000.
           To get the units in the S/T equation right, how much do we need to
           scale the bvalues provided.

        sub_sample: int or array of ints.
           If we want to sub-sample the DWI data on the sphere (in the bvecs),
           we can do one of two things: 
        1. If sub_sample is an integer, that number of random bvecs will be
           chosen from the data.

        2. If an array of indices is provided, these will serve as indices into
        the last dimension of the data and only that part of the data will be
        used
        
        """
        # If you provided file-names and not a DWI class object, we will
        # generate one for you right here and replace it inplace: 
        if DWI.__class__ in [list, np.ndarray, tuple]:
            DWI = dwi.DWI(DWI[0], DWI[1], DWI[2])
        
        self.data = DWI.data
        self.bvecs = DWI.bvecs
        
        # This factor makes sure that we have the right units for the way we
        # calculate the ADC: 
        self.bvals = DWI.bvals/scaling_factor

        # Get the inverse of the DWI affine, which xforms from fiber
        # coordinates (which are in xyz) to image coordinates (which are in ijk):
        self.affine = DWI.affine.getI()

        if sub_sample is not None:
            if np.iterable(sub_sample):
                idx = sub_sample
            else:
                idx = boot.subsample(self.bvecs[:,self.b_idx].T, sub_sample)[1]

            self.b_idx = self.b_idx[idx]
            # At this point, S_weighted will be taken according to these
            # sub-sampled indices:
            self.data = np.concatenate([self.S_weighted,
                                   self.data[:,:,:,self.b0_idx]],-1)

            self.b0_idx = np.arange(len(self.b0_idx))

            self.bvecs = np.concatenate([np.zeros((3,len(self.b0_idx))),
                                        self.bvecs[:, self.b_idx]],-1)

            self.bvals = np.concatenate([np.zeros(len(self.b0_idx)),
                                         self.bvals[self.b_idx]])
            self.b_idx = np.arange(len(self.b0_idx), len(self.b0_idx) + len(idx))
            
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
    def S_weighted(self):
        """
        The signal in b-weighted volumes
        """
        return self.data[...,self.b_idx].squeeze()

    def fit(self):
        """
        The pattern is that each one of the models will have a fit method that
        goes and fits the particulars of that model.

        XXX Need to consider the commonalities which can be outlined here. 
        """
        pass
    
class TensorModel(BaseModel):
    """
    A class for representing and solving a simple forward model. Just the
    diffusion tensor.
    
    """
    def __init__(self,
                 DWI,
                 scaling_factor=SCALE_FACTOR,
                 mask=None,
                 sub_sample=None,
                 file_name=None):
        """
        Parameters
        -----------
        DWI: A microtrack.dwi.DWI class instance, or a list containing: [the
        name of nifti file, from which data should be read, bvecs file, bvals
        file]

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


        file_name: A file to cache the initial tensor calculation in. If this
        file already exists, we pull the tensor fit out of it. Otherwise, we
        calculate the tensor fit and save this file with the params of the
        tensor fit. 
        
        """
        # Initialize the super-class:
        BaseModel.__init__(self,
                           DWI,
                           scaling_factor=scaling_factor,
                           sub_sample=sub_sample)

        if file_name is not None: 
            self.file_name = file_name
        else:
            # If DWI has a file-name, construct a file-name out of that: 
            if hasattr(DWI, 'data_file'):
                path, f = os.path.split(DWI.data_file)
                # Need to deal with the double-extension in '.nii.gz':
                file_parts = f.split('.')
                name = file_parts[0]
                extension = ''
                for x in file_parts[1:]:
                    extension = extension + '.' + x
                self.file_name = os.path.join(path, name + 'TensorModel' +
                                              extension)
            else:
                # Otherwise give up and make a file right here with a generic
                # name: 
                self.file_name = 'DTI.nii.gz'
        
        
        if mask is not None:
            self.mask = mask
        else:
            # Include everything:
            self.mask = np.ones(self.data.shape)
            
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
        if os.path.isfile(self.file_name):
            return ni.load(self.file_name).get_data()
        else: 
            mp = dti.Tensor(self.data,
                            self.bvals,
                            self.bvecs.T,
                            self.mask).model_params 

            # Save the params for future use: 
            params_ni = ni.Nifti1Image(mp, self.affine)
            params_ni.to_filename(self.file_name)
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
        lambda_1 = self.evals[..., 0]
        lambda_2 = self.evals[..., 1]
        lambda_3 = self.evals[..., 2]
        if has_ne:
            # Use numexpr to evaluate this quickly:
            fa = np.sqrt(0.5) * np.sqrt(numexpr.evaluate("(lambda_1 - lambda_2)**2 + (lambda_2-lambda_3)**2 + (lambda_3-lambda_1)**2 "))/np.sqrt(numexpr.evaluate("lambda_1**2 + lambda_2**2 + lambda_3**2"))
        else:
            fa =  np.sqrt(0.5) * np.sqrt((lambda_1 - lambda_2)**2 + (lambda_2-lambda_3)**2 + (lambda_3-lambda_1)**2 )/np.sqrt(lambda_1**2 + lambda_2**2 + lambda_3**2)

        return fa

    @desc.auto_attr
    def radial_diffusivity(self):
        return np.mean(self.evals[...,1:],-1)

    @desc.auto_attr
    def axial_diffusivity(self):
        return np.mean(self.evals[...,0],-1)

    # Self Diffusion Tensor, taken from dipy.reconst.dti:
    @desc.auto_attr
    def tensors(self):
        evals = self.evals
        evecs = self.evecs
        evals_flat = evals.reshape((-1, 3))
        evecs_flat = evecs.reshape((-1, 3, 3))
        D_flat = np.empty(evecs_flat.shape)
        for ii in xrange(len(D_flat)):
            Q = evecs_flat[ii]
            L = evals_flat[ii]
            D_flat[ii] = np.dot(Q*L, Q.T)
        return D_flat.reshape(evecs.shape)


    @desc.auto_attr
    def apparent_diffusion_coef(self):
        adc_flat = np.empty((np.prod(self.evecs.shape[:3]), len(self.b_idx)))
        tensors_flat = self.tensors.reshape((-1,3,3))
        for ii in xrange(len(adc_flat)):
            adc_flat[ii] = mtt.apparent_diffusion_coef(
                                        self.bvecs[:,self.b_idx],
                                        tensors_flat[ii])
        return adc_flat.reshape(self.evecs.shape[:3] + (len(self.b_idx),))

    @desc.auto_attr
    def fit(self):
        adc_flat = self.apparent_diffusion_coef.reshape((-1, len(self.b_idx)))
        s0_flat = self.S0.ravel()
        sig_flat = np.empty((np.prod(self.data.shape[:3]), len(self.b_idx),))

        for ii in xrange(len(sig_flat)):
            sig_flat[ii] = mtt.stejskal_tanner(s0_flat[ii],
                                               self.bvals[:, self.b_idx],
                                               adc_flat[ii])

        return sig_flat.reshape(self.data.shape[:3] + (len(self.b_idx),))


    @desc.auto_attr
    def residual(self):
        """
        The prediction-subtracted residual
        """
        return self.S_weighted - self.fit


class SphericalHarmonicsModel(BaseModel):
    """
    A class for calculating and evaluating spherical harmonic models
    
    """ 

    def __init__():
        """
        """
        raise NotImplementedError
        
    
class FiberModel(BaseModel):
    """
    A class for representing and solving microtrack models
    """
    def __init__(self,
                 DWI,
                 FG,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 scaling_factor=SCALE_FACTOR):
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
                             DWI,
                             scaling_factor=scaling_factor)

        self.axial_diffusivity = axial_diffusivity
        self.radial_diffusivity = radial_diffusivity

        # The only additional thing is that this one also has a fiber group,
        # which is xformed to match the coordinates of the DWI:
        self.FG = FG.xform(self.affine, inplace=False)


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
    def sig(self):
        """
        The signal in the voxels corresponding to where the fibers pass through.
        """ 
        return self.S_weighted[self.fg_idx_unique[0],
                               self.fg_idx_unique[1],
                               self.fg_idx_unique[2]].ravel()
