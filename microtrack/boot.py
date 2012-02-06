"""

Utilities for sub-sampling b vectors from dwi experiments 

""" 

import numpy as np
import scipy
from scipy import linalg as la

import microtrack as mt

def subsample(bvecs, n, elec_points=None):
    """

    Generate a sub-sample of size n of directions from the provided bvecs

    Parameters
    ----------
    bvecs: int array (n by 3), a set of cartesian coordinates for a set of
    bvecs 
    n: int, how many bvecs to sub-sample from this set. 
    elec_points: optional, a set of points read from the camino points, using
    Jones (2003) algorithm for electro-static repulsion
    
    Returns 
    -------
    x,y,z: The coordinates of the sub-sample
 
    Notes
    -----
    Directions are chosen from the camino-generated electro-static repulsion
    points in the directory camino_pts.

    """
    if elec_points is None: 
        mt_path = mt.__path__[0]
        e_points = np.loadtxt('%s/camino_pts/Elec%03d.txt'%(mt_path, n))

    else:
        e_points = elec_points.copy()
        
    # The very first one is n and the rest need to be reshaped as thus: 
    assert(e_points[0]==n)
    xyz = e_points[1:].reshape(e_points[1:].shape[0]/3, 3)

    # Since the camino points cover only a hemi-sphere, a random half of the
    # points need to be inverted by 180 degrees to account for potential
    # gradient asymmetries:

    # Get indices to a random half of these xyz values: 
    rand_half_idx = np.random.permutation(xyz.shape[0])[::2]
    # and turn them 180 degrees: 
    xyz[rand_half_idx] *=-1
    
    # The seed bvec is the one relative to which all the rest are chosen:
    seed = np.ceil(np.random.rand() * xyz.shape[0]).astype(int)
    seed_coords = bvecs[seed]

    # Get the rotation matrix: 
    rot = calculate_rotation(seed_coords, xyz[0])

    # Rotate all the points to align with the seed: 
    new_points = np.array(bvecs * rot)

    sample_bvecs = np.zeros((3,n))
    bvec_idx = []
    
    for vec in xrange(n):
        this = new_points[vec]
        delta = np.zeros(bvecs.shape[0])
        for j in xrange(bvecs.shape[0]):
            delta[j] = vector_angle(this, bvecs[j])

        this_idx = np.where(delta==np.min(delta))
        
        bvec_idx.append(this_idx)    
        sample_bvecs[:,vec]= bvecs[this_idx]

    return sample_bvecs, np.array(bvec_idx).squeeze()
        
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
