"""

Utilities for sub-sampling b vectors from dwi experiments 

""" 

import numpy as np
import scipy.linalg as la

import osmosis as oz
import osmosis.utils as ozu

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
        # We need a n by 3 here:
        xyz = ozu.get_camino_pts(n_dirs).T
    else:
        xyz = elec_points.copy()
            
    # Rotate all the points to align with the seed, the bvec relative to which
    # all the rest are chosen (lots going on in this one line):
    rot_to_first = ozu.calculate_rotation(
                              bvecs[:, np.ceil(np.random.randint(xyz.shape[0]))],
                              xyz[0])

    new_points = np.dot(rot_to_first, bvecs).T

    sample_bvecs = np.zeros((3, n_dirs))
    bvec_idx = []

    potential_indices = np.arange(bvecs.shape[-1])
    for vec in xrange(n_dirs):
        this = new_points[vec]
        delta = np.zeros(potential_indices.shape)
        for j in range(delta.shape[0]):
            delta[j] = ozu.vector_angle(this, bvecs[:, j])

        this_idx = np.where(delta==np.min(delta))
        
        bvec_idx.append(potential_indices[this_idx])    
        sample_bvecs[:, vec] = np.squeeze(bvecs[:, this_idx])

        # Remove bvecs that you've used, so that you don't have them more than
        # once: 
        bvecs = np.hstack([bvecs[:, :this_idx[0]],bvecs[:, this_idx[0]+1:]])
        potential_indices = np.hstack([potential_indices[:this_idx[0]],
                                            potential_indices[this_idx[0]+1:]])
        
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
    mean_principal_eigvec = np.mean(dyad,0)[0]

    theta = []
    for this_d in dyad:
        # Using a variation on equation 3 in Jones(2003), taking the arccos of
        # the correlation between the two vectors (this should yield 90 degrees
        # for orthogonal vectors and 0 degrees for identical vectors): 
        corr = (np.dot(this_d[0],mean_principal_eigvec)/
                        (np.sqrt(np.dot(this_d[0],this_d[0]) *
                               np.dot(mean_principal_eigvec,
                                      mean_principal_eigvec))))

        # We sometimes get floating point error leading to larger-than-1
        # correlation, we treat that as though it were equal to 1:
        if corr>1.0:
            corr=1.0
        angle = np.arccos(corr)

        theta.append(angle)

    # Average over all the tensors:
    return np.mean(theta)
