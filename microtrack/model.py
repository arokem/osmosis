"""

This module is used to construct and solve models of diffusion data 

"""
import numpy as np
# Import stuff for sparse matrices:
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import descriptors as desc
import microtrack.fibers as mtf
import microtrack.tensor as mtt
import microtrack.dwi as dwi

class Model(object):
    """
    A class for representing and solving microtrack models
    """
    def __init__(DWI,
                 FG,
                 axial_diffusivity,
                 radial_diffusivity):
        """
        Parameters
        ----------
        DWI: A microtrack.dwi.DWI object, or a list containing: [the name of
             nifti file, from which data should be read, bvecs file, bvals file]
        
        FG: a microtrack.fibers.FiberGroup object, or the name of a pdb file
            containing the fibers to be read in using mtf.fg_from_pdb

        """ 

        # If you provided file-names and not a DWI class object, we will
        # generate one for you right here and replace it inplace: 
        if DWI.__class__ in [list, np.ndarray, tuple]:
            DWI = dwi.DWI(DWI[0], DWI[1], DWI[2])
        
        self.data = DWI.data
        self.bvecs = DWI.bvecs
        self.bvals = DWI.bvals
        self.FG = FG
        self.axial_diffusivity = axial_diffusivity
        self.radial_diffusivity = radial_diffusivity
        
    @desc.auto_attr
    def S0(self):
        """
        Extract and average the signal for volumes in which no b weighting was
        used (b0 scans)
        """
        b0_idx = np.where(self.bvals==0)
        return self.data[...,b0_idx]
        
        
    @desc.auto_attr
    def S_weighted(self):
        """
        The signal in b-weighted volumes
        """
        b_idx = np.where(self.bvals > 0)
        return self.data[...,b_idx]

    
    @desc.auto_attr
    def matrix(self):
        """
        
        """
        pred_sig = np.zeros(self.data.shape)
        for fiber in self.FG.fibers:
            idx = fiber.coords.astype(int)
            pred_sig[idx] += fiber.predicted_signal(self.axial_diffusivity,
                                                    self.radial_diffusivity,
                                                    self.S0,
                                                    self.bvecs,
                                                    self.bvals)
