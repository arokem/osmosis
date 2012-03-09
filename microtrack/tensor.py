import warnings

import numpy as np
import scipy.linalg as la

import microtrack.descriptors as desc
import microtrack.utils as mtu


class Tensor(object):
    """
    Represent a diffusion tensor.
    """

    def __init__(self, Q, bvecs, bvals):
        """
        Initialize a Tensor object

        Parameters
        ----------
        Q: the quadratic form of the tensor. This can be one of the following:
           - A 3,3 ndarray or matrix
           - An array of length 9
           - An array of length 6

           Recall that the quadratic form has six independent parameters,
           because it is symmetric across the main diagonal, so if only 6
           parameters are provided, you can recover the other ones from that.
           For length 6

        bvecs: 3 by n array
            unit vectors on the sphere used in acquiring DWI data. 

        bvals: an array with n items
            The b-weighting used to measure the DWI data.
        
        Note
        ----

        When converting from a dt6 (length 6 array), we follow the conventions
        used in vistasoft, dt6to33:
        
        >> aa = dti6to33([0,1,2,3,4,5])

        aa(:,:,1) =
        0     3     4
        aa(:,:,2) =
        3     1     5
        aa(:,:,3) =
        4     5     2

        """

        # Cast into array, so that you can look at the shape:
        Q = np.asarray(Q)
        # Check the input:
        if Q.squeeze().shape==(9,):
            self.Q = np.matrix(np.reshape(Q, (3,3)))                
        elif Q.shape == (3,3):
            self.Q = np.matrix(Q)

        elif Q.squeeze().shape==(6,):
            # We're dealing with a dt6 of sorts:
            tmp = np.zeros((3,3))
            tmp[0, :] = [Q[0], Q[3], Q[4]]
            tmp[1, :] = [Q[3], Q[1], Q[5]]
            tmp[2, :] = [Q[4], Q[5], Q[2]]
            self.Q = np.matrix(tmp)
            
        else:
            e_s = "Q had shape: ("
            e_s += ''.join(["%s, "%n for n in Q.shape])
            e_s += "), but should have shape (9) or (3,3)"
            raise ValueError(e_s)
        
        # It needs to be symmetrical
        if not np.allclose(self.Q.T, self.Q): # Within some tolerance...
            e_s = "Please provide symmetrical Q"
            raise ValueError(e_s)

        # Check inputs:
        bvecs = np.asarray(bvecs)
        if len(bvecs.shape)>2 or bvecs.shape[0]!=3:
            e_s = "bvecs input has shape ("
            e_s += ''.join(["%s, "%n for n in bvecs.shape])
            e_s += "); please reshape to be 3 by n"
            raise ValueError(e_s)

        # They should all be unit-length (to within tolerance):
        for bv in bvecs.T:
            norm = np.sqrt(np.dot(bv,bv))
            if not np.allclose(norm, 1):
                e_s = "One of the bvecs is length %s "%norm
                e_s += "; make sure they're all approximately"
                e_s += " unit length"
                raise ValueError(e_s)
        
        bvals = np.asarray(bvals)
        if bvecs.shape[-1] != bvals.shape[0]:
            e_s = "bvecs input has shape ("
            e_s += ''.join(["%s, "%n for n in bvecs.shape])
            e_s += ") and bvals input has shape: (%s)"%bvals.shape[0]
            e_s += "; please reshape so they both have the same "
            raise ValueError(e_s)

        self.bvecs = bvecs
        self.bvals = bvals
        
    @desc.auto_attr
    def ADC(self):
        """
        Calculate the Apparent diffusion coefficient for the tensor
            
        Notes
        -----
        This is calculated as $ADC = \vec{b} Q \vec{b}^T$
        """
        # Make sure it's a matrix:
        return apparent_diffusion_coef(self.bvecs, self.Q)

    def predicted_signal(self, S0):
        """
        Calculate the signal predicted from the properties of this tensor. 
        
        Parameters
        ----------
        S0: float
            The baseline signal, relative to which the signal is calculated.
            
        bvecs: n by 3 array
            Unit vectors on the sphere for which the calculation is performed.

        bvals: float, or a len(n) array
            The b-weighting values used in each of the directions in bvecs.

        Returns
        -------
        The predicted signal in the directions given by bvecs
        
        Notes
        -----
        This implements the following formulation of the Stejskal/Tanner
        equation (see
        http://en.wikipedia.org/wiki/Diffusion_MRI#Diffusion_imaging):
        
        .. math::
        
            S = S0 exp^{-bval ADC }
    
        Where ADC is:

        .. math::
        
            ADC = \vec{b} Q \vec{b}^T
        
        """

        # We calculate ADC and pass that to the S/T equation:
        return stejskal_tanner(S0, self.bvals, self.ADC)

    @desc.auto_attr
    def decompose(self):
        return mtu.decompose_tensor(self.Q)

    
    def convolve_odf(self, odf, S0):
        """

        Convolve an orientation distribution function with the predicted signal
        from the Tensor

        Parameters
        ----------
        odf: 1-d array
            An estimate of the orientation distribution function in a given voxel
            in each bvec

        S0: The signal in non diffusion weighted scans

        Returns
        -------
        signal estimate.

        """
        tensor_signals = []

        for idx, bvec in enumerate(self.bvecs.T):
            # This can be precomputed for this tensor, without knowledge of a
            # specific S0:
            rot_tensor = self._rotations[idx]
            # Add the signal for this tensor, weighted by the ODF at that point:
            tensor_signals.append(odf[idx] * rot_tensor.predicted_signal(S0))
            
        signal = np.sum(tensor_signals,0)

        return signal

    
    @desc.auto_attr
    def _rotations(self):
        """
        Cache the rotated versions of the tensor, which can be reused to
        predict the signal for different values of S0
        """
        rot_tensors = []
        # Decompose the tensor:
        evals, evecs = self.decompose
        e1 = evecs[0]
        for idx, bvec in enumerate(self.bvecs.T):
            rot_tensor_e = evecs * mtu.calculate_rotation(bvec, e1)
            # Now create the same tensor, just with rotated eigen-vectors:
            rot_tensors.append(tensor_from_eigs(rot_tensor_e,
                                                evals, self.bvecs, self.bvals))
            
        return rot_tensors

    
def stejskal_tanner(S0, bvals, ADC):
    """
    Parameters
    ----------
    S0: float
       The signal observed in the voxel with 0 b-weighting (baseline).

    bvals: float, or a len(n) array
       The b-weighting values used in each of the directions in bvecs.

    ADC: float
        The apparent diffusion coefficient in the order of the relevant bvecs.
    
    
    Notes
    -----
    
    This is an implementation of the Stejskal Tanner equation, in the following
    form: 
    
    .. math::
    
    S = S0 exp^{-bval ADC}

    Where $ADC = (\vec{b}*Q*\vec{b}^T)$

    """

    e = np.exp(-np.asarray(bvals) * ADC)
    return np.asarray(S0) * e.T 
    

def apparent_diffusion_coef(bvecs, q):
    """
    $ADC = \vec{b} Q \vec{b}^T$
    """
    bvecs = np.matrix(bvecs)
    return np.diag(bvecs.T* q* bvecs)


def tensor_from_eigs(evecs, evals, bvecs, bvals):
    """
    Create a tensor from an eigen-vector/eigen-value combination, instead of
    from the q-form

    Parameters 
    """

    return Tensor(mtu.tensor_from_eigs(evals, evecs), bvecs, bvals)
