"""
A variety of utility functions

"""

import numpy as np
import scipy
import scipy.linalg as la


def unique_rows(in_array, dtype='f4'): 
    """
    This (quickly) finds the unique rows in an array

    Parameters
    ----------
    in_array: ndarray
        The array for which the unique rows should be found

    dtype: str, optional
        This determines the intermediate representation used for the
        values. Should at least preserve the values of the input array.

    Returns
    -------
    u_return: ndarray
       Array with the unique rows of the original array.
    
    """
    x = np.array([tuple(in_array.T[:,i]) for i in
                  xrange(in_array.shape[0])],
        dtype=(''.join(['%s,'%dtype]* in_array.shape[-1])[:-1])) 

    u,i = np.unique(x,return_index=True)
    u_i = x[np.sort(i)]
    u_return = np.empty((in_array.shape[-1],len(u_i)))
    for j in xrange(len(u_i)):
        u_return[:,j] = np.array([x for x in u_i[j]])

    # Return back the same dtype as you originally had:
    return u_return.T.astype(in_array.dtype)
        
def l2_norm(x):
    """
    The l2 norm of a vector x is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    return np.sqrt(np.dot(x,x))


def unit_vector(x, norm=l2_norm):
    """
    Normalize the vector by a norm function (defaults to the l2_norm)
    """
    return x/l2_norm(x)


def null_space(A, eps=1e-15):
    """
    Calculate the nullspace of the matrix A    
    """ 
    u, s, vh = la.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def quat2rot(w,x,y,z):
    """
    Given a quaternion, defined by w,x,y,z, return the rotation matrix
    """
    return np.matrix(np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y],
                                 [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x],
                                [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]]))

def vector_angle(a,b):
    """
    Calculate the angle between two vectors. 
    """
    # Normalize both to unit length:
    norm_a = unit_vector(a)
    norm_b = unit_vector(b)

    # If the vectors are identical, the anlge is 0 per definition: 
    if np.all(norm_a==norm_b):
        return 0
    else: 
        return np.arccos(np.dot(norm_a,norm_b))

def calculate_rotation(a,b):
    """
    Calculate the rotation matrix to rotate from vector a to vector b.
    """
    alpha = vector_angle(a,b)

    # The rotation between identical vectors is the identity: 
    if alpha == 0:
        return np.eye(3)

    # Otherwise, we need to do some math: 
    # Find the orthonormal basis for the null-space of these two vectors:
    u = null_space(np.matrix([a,b,[0,0,0]]))

    # Using quaternion notation (See
    # http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#From_the_rotations_to_the_quaternions): 
    
    w = np.cos(alpha/2); 
    xyz = u * np.sin(alpha/2); 

    rot = quat2rot(w,xyz[0], xyz[1], xyz[2])

    # This is accurate up to a sign reversal in each coordinate, so we rotate
    # back and compare to the original:

    rot_back = b * rot
    sign_reverser = np.sign(np.sign(rot_back == np.sign(a)) - 0.5)

    # Multiply each line by it's reverser and reassmble the matrix:
    return np.matrix(np.array([np.array(rot[i,:]) *
                            sign_reverser[0,i] for i in range(3)]).squeeze())

def calculate_rotation(a,b):
    """
    Calculate the rotation matrix to rotate from vector a to vector b.
    """
    alpha = vector_angle(a,b)

    # The rotation between identical vectors is the identity: 
    if alpha == 0:
        return np.eye(3)

    # Otherwise, we need to do some math: 
    # Find the orthonormal basis for the null-space of these two vectors:
    u = null_space(np.matrix([a,b,[0,0,0]]))

    # Using quaternion notation (See
    # http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#From_the_rotations_to_the_quaternions): 
    
    w = np.cos(alpha/2); 
    xyz = u * np.sin(alpha/2); 

    rot = quat2rot(w,xyz[0], xyz[1], xyz[2])

    # This is accurate up to a sign reversal in each coordinate, so we rotate
    # back and compare to the original:

    rot_back = b * rot
    sign_reverser = np.sign(np.sign(rot_back == np.sign(a)) - 0.5)

    # Multiply each line by it's reverser and reassmble the matrix:
    return np.matrix(np.array([np.array(rot[i,:]) *
                            sign_reverser[0,i] for i in range(3)]).squeeze())

