""""
Clustering algorithms
=====================

Spherical k means (spkm) and online spherical k means (ospkm), based on: 

http://www.sci.utah.edu/~weiliu/research/clustering_fmri/Zhong_sphericalKmeans.pdf

"""

import numpy as np
import dipy.core.geometry as geo

import osmosis.utils as ozu

def spkm(data, k, seeds=None, antipodal=True):
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

   antipodal : bool
      In cases in which antipodal symmetery can be assumed, we want to cluster
      together points that are pointing close to *opposite* directions. In that
      case, correlations between putative centroids and each data point treats
      correlation and anti-correlation in equal vein.
    
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
      
   mu = seeds
   is_changing = True
   last_y_n = False
   while is_changing:
      # 2. Data assignment:
      # Calculate all the correlations in one swoop:
      corr = np.dot(data, seeds.T)
      # In cases where antipodal symmetry is assumed, 
      if antipodal==True:
         corr = np.abs(corr)
      
      # This chooses the centroid for each one:
      y_n = np.argmax(corr, -1)
      # 3. Centroid estimation:
      for this_k in range(k):
         idx = np.where(y_n==this_k)
         this_sum = np.sum(data[idx], axis=0)
         # This goes into the volume of the sphere and then renormalizes to the
         # surface:
         mu[this_k] = this_sum / ozu.l2_norm(this_sum)

      # Did it change?
      if np.all(y_n == last_y_n):
         # 4. Stop if there's no change in assignment:
         is_changing = False
      else:
         last_y_n = y_n

   return mu, y_n
    
def ospkm():
    """
    Online spherical k means
    """
    pass

