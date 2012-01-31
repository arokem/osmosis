import numpy as np
import scipy.linalg as la

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
        if not np.all(self.Q.T == self.Q):
            e_s = "Please provide symmetrical Q"
            raise ValueError(e_s)

    def ADC(self,bvecs):
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
    Produce the tensors for a given a fiber.
    
    """
    grad = np.array(np.gradient(fiber.coords)[0])
    u,s,v = la.svd(grad)
    Q = (np.matrix(u).T *
         np.matrix(np.diag([axial_diffusivity,
                            radial_diffusivity,
                            radial_diffusivity])) *
            np.matrix(u))

    return Tensor(Q)

def fiber_signal(tensor_list, ):

    """
    Compute the fiber contribution to the signal, based on a simplified
    Stejskal/Tanner equation.

    Parameters
    ----------
    """
    raise NotImplementedError
