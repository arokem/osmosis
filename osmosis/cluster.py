""""
Clustering algorithms
=====================

Spherical k means (spkm) and online spherical k means (ospkm), based on: 

http://www.sci.utah.edu/~weiliu/research/clustering_fmri/Zhong_sphericalKmeans.pdf

"""
from __future__ import division

import numpy as np
import dipy.core.geometry as geo

import osmosis.utils as ozu

def spkm(data, k, weights=None, seeds=None, antipodal=True, max_iter=1000,
         calc_sse=True):
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
      In cases in which antipodal symmetry can be assumed, we want to cluster
      together points that are pointing close to *opposite* directions. In that
      case, correlations between putative centroids and each data point treats
      correlation and anti-correlation in equal vein.

   max_iter : int
       If you run this many iterations without convergence, warn and exit.

   calc_sse : bool
      Whether to calculate SSE or not. 

   Returns
   -------
   mu : the estimated centroid 
   y_n : assignments of each data point to a centroid
   SSE : the sum of squared error in centroid-to-data-point assignment
   
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
      
   mu = seeds.copy()
   is_changing = True
   last_y_n = False
   iter = 0
   while is_changing:
     
      # Make sure they're all unit vectors, so that correlation below is scaled
      # properly: 
      mu = np.array([ozu.unit_vector(x) for x in mu])
      data = np.array([ozu.unit_vector(x) for x in data])

      # 2. Data assignment:
      # Calculate all the correlations in one swoop:
      corr = np.dot(data, mu.T)
      # In cases where antipodal symmetry is assumed, 
      if antipodal==True:
         corr = np.abs(corr)

      # This chooses the centroid for each one:
      y_n = np.argmax(corr, -1)
      
      # 3. Centroid estimation:
      for this_k in range(k):
         idx = np.where(y_n==this_k)
         if len(idx[0])>0: 
            # The average will be based on the data points that are considered
            # in this centroid with a weighted average: 
            this_sum = np.dot(weights[idx], data[idx])

            # This goes into the volume of the sphere, so we renormalize to the
            # surface (or to the origin, if it's 0):
            this_norm =  ozu.l2_norm(this_sum)

            if this_norm > 0:
               # Scale by the mean of the weights  
               mu[this_k] = (this_sum / this_norm) * np.mean(weights[idx]) 
            elif this_norm < 0:
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
         is_changing=False

      # Once you are done computing 'em all, calculate the resulting SSE: 
      SSE = 0
      if calc_sse: 
         for this_k in range(k):
            idx = np.where(y_n==this_k)
            len_idx = len(idx[0])
            if len_idx > 0:
               scaled_data = data[idx] * weights[idx].reshape(len_idx,1)
               SSE += np.sum( (mu[this_k] - scaled_data) ** 2)
             
   return mu, y_n, SSE

    
def ospkm():
    """
    Online spherical k means
    """
    pass


# Another thing from: http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikits-learn-k-means

import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

def kmeans(X, seeds=None, delta=.001, maxiter=10, metric="euclidean", p=2,
            verbose=False):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if seeds is None:
       centres = np.random.sample(X, k)
    else:
       centres = seeds

    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50.astype(int)
        print "kmeans: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d

def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, metric="euclidean", p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centres=, ... )
        in: either initial centres= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = kmeans(
                X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................
if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10000
    dim = 10
    ncluster = 10
    kmsample = 100  # 0: random centres, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    seed = 1

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    print "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
        N, dim, ncluster, kmsample, metric)
    X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/
    t0 = time()
    if kmsample > 0:
        centres, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcentres = randomsample( X, ncluster )
        centres, xtoc, dist = kmeans( X, randomcentres,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print "%.0f msec" % ((time() - t0) * 1000)

    # also ~/py/np/kmeans/test-kmeans.py
