""""
Clustering algorithms
=====================

Spherical k means (spkm) and online spherical k means (ospkm), based on: 

http://www.sci.utah.edu/~weiliu/research/clustering_fmri/Zhong_sphericalKmeans.pdf

"""

import numpy as np
import dipy.core.geometry as geo

def spkm(data, k, change_th=1, seeds=None):
    """
    Spherical k means. 

    Parameters
    ----------
    data : 2d - float array
        Unit vectors on the hyper-sphere. This array has n data points rows, by
        m feature columns.

    k : int
       The number of clusters

    seeds : float array (optional).
        If n by k array is provided, these are used as centroids to initialize
        the algorithm. Otherwise, random centroids are chosen

    XXX TODO: differentiate between the case in which angles go from 0-180 and
    the case in which they go from 0-90 (symmetry).
    
    """
    # 1. Initialization:
    if seeds is None:
        # Choose random seeds
        # thetas are uniform [0,pi]:
        theta = np.random.rand(k) * np.pi
        # phis are uniform [0, 2pi]
        phi = np.random.rand(k) * 2 * np.pi
        # They're all unit vectors:
        r = np.ones(k)
        # et voila:
        seeds = np.array(geo.sphere2cart(theta, phi, r)).T

    Y = np.copy(seeds)
    dY = np.inf
    while dY>change_th:
        last_Y = np.copy(Y)
        # 2. Data assignment:
        # Calculate all the correlations in one swoop:
        corr = np.dot(data, seeds.T)
        # This chooses the centroid for each one:
        y_n = np.argmax(corr, -1)
        # 3. Centroid estimation:
        for this_k in range(k):
            idx = np.where(y_n==this_k)
            print idx
            Y[this_k] = np.mean(data[idx])
        # How much did it change
        dY = np.sum(Y - last_Y)
    # 4. Stopping (we'll simply pop out of that loop and return the new
    # centroids and the data assignments):
    return Y, y_n
    
def ospkm():
    """
    Online spherical k means
    """


