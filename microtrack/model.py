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

class TensorModel(object):
    """
    A class for representing and solving a simple forward model. Just the
    diffusion tensor.
    
    """

    def __init__(self,
                 DWI,
                 axial_diffusivity=1.5,
                 radial_diffusivity=0.5,
                 scaling_factor=10000):
        """
        Parameters
        -----------

        DWI: A microtrack.dwi.DWI class instance, or a list containing: [the
        name of nifti file, from which data should be read, bvecs file, bvals
        file]

        axial_diffusivity: The axial diffusivity of a single fiber population.

        radial_diffusivity: The radial diffusivity of a single fiber population.

        scaling_factor: This scales the b value for the Stejskal/Tanner equation
        """
        if DWI.__class__ in [list, np.ndarray, tuple]:
            DWI = dwi.DWI(DWI[0], DWI[1], DWI[2])

        # If you provided file-names and not a DWI class object, we will
        # generate one for you right here and replace it inplace: 
        if DWI.__class__ in [list, np.ndarray, tuple]:
            DWI = dwi.DWI(DWI[0], DWI[1], DWI[2])
        
        self.data = DWI.data
        self.bvecs = DWI.bvecs # Transpose so that they fit the conventions
                                 # in microtrack.tensor
        
        # XXX What is this factor? 
        self.bvals = DWI.bvals/scaling_factor

        # Get the inverse of the DWI affine, which xforms from fiber
        # coordinates (which are in xyz) to image coordinates (which are in ijk):
        self.affine = DWI.affine.getI()
        self.axial_diffusivity = axial_diffusivity
        self.radial_diffusivity = radial_diffusivity
        
class FiberModel(TensorModel):
    """
    A class for representing and solving microtrack models
    """
    def __init__(self,
                 DWI,
                 FG,
                 axial_diffusivity=1.5,
                 radial_diffusivity=0.5,
                 scaling_factor=10000):
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
        TensorModel.__init__(self,
                             DWI,
                             axial_diffusivity,
                             radial_diffusivity,
                             scaling_factor)
        
        self.FG = FG.xform(self.affine, inplace=False)
        
    @desc.auto_attr
    def S0(self):
        """
        Extract and average the signal for volumes in which no b weighting was
        used (b0 scans)
        """
        b0_idx = np.where(self.bvals==0)
        return np.mean(self.data[...,b0_idx],-1).squeeze()
        
    @desc.auto_attr
    def S_weighted(self):
        """
        The signal in b-weighted volumes
        """
        b_idx = np.where(self.bvals > 0)
        return self.data[...,b_idx].squeeze()

    @desc.auto_attr
    def idx(self):
        """
        Indices into the coordinates of the fiber-group
        """
        return self.FG.coords().astype(int)
    
    @desc.auto_attr
    def matrix(self):
        """
        The matrix of fiber-contributions to the DWI signal.
        """
        b_idx = np.where(self.bvals!=0)
        pred_sig = []
        # Sum the signals from all the fibers/all the nodes in each voxel, by
        # summing over the predicted signal from each fiber through all its
        # nodes and just adding it in:
        i = 1
        for fiber in self.FG.fibers:
            print "working on fiber %s"%i
            i += 1
            this_sig = np.zeros(self.S_weighted.shape)
            idx = fiber.coords.astype(int)
            this_sig[idx[0], idx[1], idx[2], :] += \
                fiber.predicted_signal(self.axial_diffusivity,
                                       self.radial_diffusivity,
                                       self.S0[idx[0],idx[1],idx[2]],
                                       self.bvecs[:, b_idx].squeeze(),
                                       self.bvals[b_idx]).T
        
            # Bring everything to the common frame of reference: 
            pred_sig.append(this_sig[self.idx[0],
                                     self.idx[1],
                                     self.idx[2]].ravel())

        return np.array(pred_sig)

    @desc.auto_attr
    def sig(self):
        """
        The signal in the voxels corresponding to where the fibers pass through.
        """ 

        return self.S_weighted[self.idx[0], self.idx[1], self.idx[2]].ravel()
