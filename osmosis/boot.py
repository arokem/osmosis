"""

Utilities for sub-sampling b vectors from dwi experiments 

""" 

import numpy as np
import scipy.linalg as la

import osmosis as mt
import osmosis.utils as mtu

def subsample(bvecs, n_dirs, elec_points=None):
    """

    Generate a sub-sample of size n of directions from the provided bvecs

    Parameters
    ----------
    bvecs: int array (n by 3), a set of cartesian coordinates for a set of
    bvecs 
    n_dirs: int, how many bvecs to sub-sample from this set. 
    elec_points: optional, a set of points read from the camino points, using
    Jones (2003) algorithm for electro-static repulsion
    
    Returns 
    -------
    [x,y,z]: The coordinates of the sub-sample
    bvec_idx: The indices into the original bvecs that would give this
        sub-sample 
 
    Notes
    -----
    Directions are chosen from the camino-generated electro-static repulsion
    points in the directory camino_pts.

    """
    if elec_points is None: 
        mt_path = mt.__path__[0]
        e_points = np.loadtxt('%s/camino_pts/Elec%03d.txt'%(mt_path, n_dirs))

    else:
        e_points = elec_points.copy()
        
    # The very first one is n and the rest need to be reshaped as thus: 
    assert(e_points[0]==n_dirs)
    xyz = e_points[1:].reshape(e_points[1:].shape[0]/3, 3)

    # Since the camino points cover only a hemi-sphere, a random half of the
    # points need to be inverted by 180 degrees to account for potential
    # gradient asymmetries. Get indices to a random half of these xyz values
    # and turn them 180 degrees:
    xyz[np.random.permutation(xyz.shape[0])[::2]] *=-1
    
    # Rotate all the points to align with the seed, the bvec relative to which
    # all the rest are chosen (lots going on in this one line):
    new_points = np.array(bvecs *
                          mtu.calculate_rotation(
                              bvecs[np.ceil(np.random.rand() *
                                            xyz.shape[0]).astype(int)],
                              xyz[0]))

    sample_bvecs = np.zeros((3, n_dirs))
    bvec_idx = []
    
    for vec in xrange(n_dirs):
        this = new_points[vec]
        delta = np.zeros(bvecs.shape[0])
        for j in xrange(bvecs.shape[0]):
            delta[j] = mtu.vector_angle(this, bvecs[j])

        this_idx = np.where(delta==np.min(delta))
        
        bvec_idx.append(this_idx)    
        sample_bvecs[:, vec] = bvecs[this_idx]

    return sample_bvecs, np.array(bvec_idx).squeeze()
        
def dyadic_tensor(eigs,average=True):
    """
    Calculate the dyadic tensor of the principal diffusion direction from
    multiple (3,3) eigenvector sets

    Parameters
    ----------
    eigs: (n,3,3) array
        Sets of eigenvectors that correspond to the same voxel

    Notes
    -----
    The dyadic tensor is defined as:

    .. math:: 

       $ \langle \epsilon_1 \epsilon_1 \rangle_j = $

       \\( \langle ( \begin{matrix} \epsilon_{1x}^2 & \epsilon_{1x}\epsilon_{1y} & \epsilon_{1z}; && \epsilon_{1x}\epsilon_{1y} & \epsilon_{1y}^2 & \epsilon_{1y}\epsilon_{1z}; && \epsilon_{1x}\epsilon_{1z} & \epsilon_{1y}\epsilon_{1} & \epsilon_{1z}^2 \end{matrix} )\rangle \\)

      = $\frac{1}{N} \sum_{j=1}^{N} \epsilon_1^j \epsilon_1^{jT}$


    Jones (2003). Determining and visualizing uncertainty in estimates of fiber
    orientation from diffusion tensor MRI MRM, 49: 7-12
    """

    eigs = np.asarray(eigs)
    dyad = np.empty(eigs.shape)
    
    for idx in xrange(eigs.shape[0]):
        # We only look at the first eigen-vector:
        dyad[idx] = np.matrix(eigs[idx][0]).T * np.matrix(eigs[idx][0])

    if average:
        return np.mean(dyad, 0)
    else:
        return dyad

def dyad_coherence(dyad): 
    """
    A measure of the dispersion of an average dyadic tensor
    
    $\kappa = 1-\sqrt{\frac{\beta_2 + \beta_3}{\beta_1}}$

    Jones (2003). Determining and visualizing uncertainty in estimates of fiber
    orientation from diffusion tensor MRI MRM, 49: 7-12
    """
    vals, vecs = la.eigh(dyad)

    # The eigenvalues are returned in *ascending* order:
    return (1 - np.sqrt((vals[0] + vals[1])/(2*vals[2])))

def dyad_dispersion(dyad):
    """
    A measure of dispersion of the dyadic tensors of several tensors. Requires
    the full distribution of dyadic tensors (dyadic_tensor calculated with
    average=False) 
    """
    mean_eigvals, mean_eigvecs  = la.eig(np.mean(dyad,0))

    mean_principal_eigvec = mean_eigvecs[0]
                 
    theta = []
    for this_d in dyad:
        # Using equation 3 in Jones(2003):
        theta.append(np.arccos(np.dot(this_d[0], mean_principal_eigvec)))

    # Average over all the tensors:
    return np.mean(theta)
