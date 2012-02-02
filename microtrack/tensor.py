import numpy as np
import scipy.linalg as la
import descriptors as desc

class Tensor(object):
    """
    Represent a diffusion tensor.
    """

    def __init__(self, Q):
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

    def ADC(self, bvecs):
        """
        Calculate the Apparent diffusion coefficient for the tensor

        Parameters
        ----------
        bvecs: n by 3 array
            unit vectors on the sphere

        Notes
        -----
        This is calculated as $ADC = \vec{b} Q \vec{b}^T$
        """
        # Make sure it's a matrix:
        bvecs = np.matrix(bvecs)       
        return np.diag(bvecs*self.Q*bvecs.T)
    
# XXX Consider putting these somewhere else, to separate concerns (fibers and
# tensors...) ? 
def fiber_tensors(fiber, axial_diffusivity, radial_diffusivity):
    """
    Produce the tensors for a given a fiber, along its coords.

    Parameters
    ----------
    fiber: A Fiber object.

    axial_diffusivity: float
         The estimated axial diffusivity of a single fiber tensor. 

    radial_diffusivity: float
         The estimated radial diffusivity of a single fiber tensor.
         
    Note
    ----
    Estimates of the radial/axial diffusivities may rely on empirical
    measurements (for example, the AD in the Corpus Callosum), or may be based
    on a biophysical model of some kind. 
    """
    tensors = []

    grad = np.array(np.gradient(fiber.coords)[1])
    for this_grad in grad.T:
        u,s,v = la.svd(np.matrix(this_grad.T))
        this_Q = (np.matrix(v) *
             np.matrix(np.diag([axial_diffusivity,
                                radial_diffusivity,
                                radial_diffusivity])) *
            np.matrix(v).T)
        tensors.append(Tensor(this_Q))
    return tensors

def fiber_signal(S0, bvecs, bvals, tensor_list):
    """
    Compute the fiber contribution to the signal in a particular voxel,
    based on a simplified Stejskal/Tanner equation:

    .. math::
    
       S = S0 exp^{-bval (\vec{b}*Q*\vec{b}^T)}

    Where $\vec{b} * Q * \vec{b}^t$ is the ADC for each tensor
    

    Parameters
    ----------
    """
    n_tensors = len(tensor_list)
    
    # Calculate all the ADCs and hold them in an ordered list: 
    ADC = []
    for T in tensor_list:
        ADC.append(T.ADC(bvecs))

    # Cast as array:
    ADC = np.asarray(ADC)

    # This is the equation itself: 
    S = (S0 *
         np.exp(
             -1 * np.tile(bvals, ADC.shape[0]) * ADC.ravel()
             ).reshape(ADC.shape))

    # We sum over all the niodes in the voxel:
    return np.sum(S,0)
