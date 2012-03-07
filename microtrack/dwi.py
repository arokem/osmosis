"""

A module for representing diffusion weighted imaging data

"""
import warnings

# We want to try importing numexpr for some array computations, but we can do
# without:
try:
    import numexpr
    has_numexpr = True
except ImportError: 
    has_numexpr = False

import scipy.stats as stats
import numpy as np
import nibabel as ni

import microtrack.descriptors as desc

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
            
        # You can provide your own affine, if you want and that bypasses the
        # class method provided below as an auto-attr:
        if affine is not None:
            self.affine = np.matrix(affine)

        # If a mask is provided, we will use it to access the data
        if mask is not None:
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
