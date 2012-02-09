"""

Utilities for sub-sampling b vectors from dwi experiments 

""" 

import numpy as np

import microtrack as mt
import microtrack.utils as mtu

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
    x,y,z: The coordinates of the sub-sample
 
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
        

    
