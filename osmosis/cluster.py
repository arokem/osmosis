""""
Clustering algorithms
=====================

Spherical k means (spkm) and online spherical k means (ospkm), based on: 

http://www.sci.utah.edu/~weiliu/research/clustering_fmri/Zhong_sphericalKmeans.pdf

"""

import numpy as np
import dipy.core.geometry as geo

import osmosis.utils as ozu

def spkm(data, k, weights=None, seeds=None, antipodal=True, max_iter=100):
   """
   Spherical k means. 

   Parameters
   ----------
   data : 2d float array
        Unit vectors on the hyper-sphere. This array has n data points rows, by
        m feature columns.

   k : int
       The number of clusters

   weights : 1d float array 
       Some data-points may be more important than others, so they will receive
       more weighting in determining the centroids 

   seeds : float array (optional).
        If n by k array is provided, these are used as centroids to initialize
        the algorithm. Otherwise, random centroids are chosen

   antipodal : bool
      In cases in which antipodal symmetery can be assumed, we want to cluster
      together points that are pointing close to *opposite* directions. In that
      case, correlations between putative centroids and each data point treats
      correlation and anti-correlation in equal vein.

   max_iter : int
       If you run this many iterations without convergence, warn and exit.

   Returns
   -------
   mu : the estimated centroid 
   y_n : assignments of each data point to a centroid
   corr : the correlation between the centroids and the data
   """
   # 0. Preliminaries:
   # For the calculation of the centroids, we want to make sure that the data
   # are all pointing into the same hemisphere (expects 3 by n):
   data = ozu.vecs2hemi(data.T).T
   # If no weights are provided treat all data points equally:
   if weights is None:
      weights = np.ones(data.shape[0])
   
   # 1. Initialization:
   if seeds is None:
      # Choose random seeds.
      # thetas are uniform [0,pi]:
      theta = np.random.rand(k) * np.pi
      # phis are uniform [0, 2pi]
      phi = np.random.rand(k) * 2 * np.pi
      # They're all unit vectors:
      r = np.ones(k)
      # et voila:
      seeds = np.array(geo.sphere2cart(theta, phi, r)).T
      
   mu = seeds
   is_changing = True
   last_y_n = False
   iter = 0
   while is_changing:
      # 2. Data assignment:
      # Calculate all the correlations in one swoop:
      corr = np.dot(data, mu.T)
      # In cases where antipodal symmetry is assumed, 
      if antipodal==True:
         corr = np.abs(corr)
      # This chooses the centroid for each one:
      y_n = np.argmax(corr, -1)
      # And this one keeps a weighted average of the highest value for each
      # data-point (this is a proxy of a goodness of fit):
      gof = np.dot(weights, np.max(corr, -1)) / np.sum(weights)

      # 3. Centroid estimation:
      for this_k in range(k):
         idx = np.where(y_n==this_k)
         # The average will be based on a weighted sum of the data points that
         # are considered in this centroid: 
         this_sum = np.dot(weights[idx], data[idx])
         # This goes into the volume of the sphere and then renormalizes to the
         # surface (or to the origin):
         this_norm =  ozu.l2_norm(this_sum)
         if this_norm > 0: 
            mu[this_k] = this_sum / this_norm
         else:
            mu[this_k] = np.array([0,0,0])

      # Did it change?
      if np.all(y_n == last_y_n):
         # 4. Stop if there's no change in assignment:
         is_changing = False
      else:
         last_y_n = y_n
         
      # Another stopping condition is if this has gone on for a while 
      iter += 1
      if iter>max_iter:
         break
      
   return mu, y_n, gof
    
def ospkm():
    """
    Online spherical k means
    """
    pass

