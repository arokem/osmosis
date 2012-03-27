"""
A variety of utility functions

"""

import numpy as np
import scipy
import scipy.linalg as la

# We want to try importing numexpr for some array computations, but we can do
# without:
try:
    import numexpr
    has_numexpr = True
except ImportError: 
    has_numexpr = False


def intersect(arr_list):
    """
    Return the values that are in all arrays.

    Parameters
    ----------
    arr_list: list
        list of ndarrays.
    

    Returns
    -------
    arr: 1-d array with all the values.

    """
    
    arr = arr_list[0].ravel()
    for this_arr in arr_list[1:]:
        arr = np.intersect1d(arr, this_arr.ravel())

    return arr

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

    u,i = np.unique(x, return_index=True)
    u_i = x[np.sort(i)]
    u_return = np.empty((in_array.shape[-1],len(u_i)))
    for j in xrange(len(u_i)):
        u_return[:,j] = np.array([x for x in u_i[j]])

    # Return back the same dtype as you originally had:
    return u_return.T.astype(in_array.dtype)
        
def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    return np.sqrt(np.dot(arr, arr))


def unit_vector(arr, norm_func=l2_norm):
    """
    Normalize the array by a norm function (defaults to the l2_norm)

    Parameters
    ----------
    arr: ndarray

    norm_func: callable
       A function that is used here to normalize the array.

    Note
    ----
    The norm_func is the thing that defines what a unit vector is for the
    particular application.
    
    """
    return arr/norm_func(arr)


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
    sign_reverser = np.sign((np.sign(rot_back) == np.sign(a)) - 0.5)

    # Multiply each line by it's reverser and reassmble the matrix:
    return np.matrix(np.array([np.array(rot[i,:]) *
                            sign_reverser[0,i] for i in range(3)]).squeeze())


def fractional_anisotropy(lambda_1, lambda_2, lambda_3):
    """
    .. math::

            FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                        \lambda_3)^2+(\lambda_2-lambda_3)^2}{\lambda_1^2+
                        \lambda_2^2+\lambda_3^2} }

    """ 
    if has_numexpr:
        # Use numexpr to evaluate this quickly:
        fa = np.sqrt(0.5) * np.sqrt(numexpr.evaluate("(lambda_1 - lambda_2)**2 + (lambda_2-lambda_3)**2 + (lambda_3-lambda_1)**2 "))/np.sqrt(numexpr.evaluate("lambda_1**2 + lambda_2**2 + lambda_3**2"))

    else:
        fa =  np.sqrt(0.5) * np.sqrt((lambda_1 - lambda_2)**2 + (lambda_2-lambda_3)**2 + (lambda_3-lambda_1)**2 )/np.sqrt(lambda_1**2 + lambda_2**2 + lambda_3**2)

    return fa

def ols_matrix(A, norm=None):
    """
    Generate the matrix used to solve OLS regression.

    Parameters
    ----------

    A: float array
        The design matrix

    norm: callable, optional
        A normalization function to apply to the matrix, before extracting the
        OLS matrix.

    Notes
    -----

    The matrix needed for OLS regression for the equation:

    ..math ::

        y = A \beta

   is given by:

    ..math ::

        \hat{\beta} = (A' x A)^{-1} A' y

    See also
    --------
    http://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation
    """

    A = np.asarray(A)
    
    if norm is not None:
        X = norm(np.matrix(A.copy()))
    else:
        X = np.matrix(A.copy())

    return la.pinv(X.T * X) * X.T


def cls_matrix():
    """
    Constrained least squares estimation. Compute the parameters $\hat{\beta}$
    that fulfill the constraint: $Q\beta=c$
    
    http://en.wikipedia.org/wiki/Ordinary_least_squares#Constrained_estimation
    
    """
    # XXX Make it! 
    raise NotImplementedError

def decompose_tensor(tensor, non_negative=True):
    """
    Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors of self-diffusion tensor. (Basser et al., 1994a)

    Parameters
    ----------
    D : array (3,3)
        array holding a tensor. Assumes D has units on order of
        ~ 10^-4 mm^2/s

    Returns
    -------
    eigvals : array (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (3,3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Note that in the output of la.eigh, eigenvectors are columnar
        (e.g. eigvecs[:,j] is associated with eigvals[j])

    See Also
    --------
    numpy.linalg.eig
    """

    #outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = la.eigh(tensor)

    #need to reorder the eigenvalues and associated eigenvectors, so that they
    #are in descending order:
    eigenvecs = eigenvecs[:,::-1].T
    eigenvals = eigenvals[::-1]

    if non_negative: 
        # Forcing negative eigenvalues to 0
        eigenvals = np.maximum(eigenvals, 0)
        # b ~ 10^3 s/mm^2 and D ~ 10^-4 mm^2/s
        # eigenvecs: each vector is columnar

    return eigenvals, eigenvecs


def tensor_from_eigs(evals, evecs):
    """
    Calculate the self diffusion tensor from evecs and evals. This is the
    inverse of decompose_tensor.

    Parameters
    ----------
    evecs: 3x3 float array, each row is an eigen-vector

    evals: 3 float arrays
    
    Returns
    -------
    The self diffusion tensor
    
    """
    evecs = np.asarray(evecs)
    return np.dot((evals*evecs.T), evecs)
    
    
def fit_tensor(bvecs, data): 
    """
    This is some potentially useful code (as in, no clear use for now) that
    fits a tensor to any arbitrary data. It undoes some of the DWI-specific
    stuff that dipy does.
    """ 

    # Construct the same design matrix you normally would (except keep it
    # positive (by setting the input bval to -1) and don't keep the last
    # column (which corresponds to the S0):
    design_matrix = dti.design_matrix(bvecs,
                                      -1 * np.ones(bvecs.shape[-1]))[:,:6]

    ols_matrix = ols_matrix(design_matrix)

    return np.array(np.dot(ols_matrix, data))

def euclidian_distance(x,y):
    """
    Compute the euclidian distances between all elements of the list x and all
    elements of the list y

    Parameters
    ----------
    x,y: lists of ndarrays. All arrays have the same dimensionality, so that
    the euclidian distance is defined for every pair of items
    
    Returns
    -------
    arr: Array where the distance between element i in x and element j in y is
        arr[i,j]

    """
    
    arr = np.empty((len(x), len(y)))
    for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[1]):
            arr[i, j] = root_ss(x[i]-y[j])
            
    return arr
    
def root_ss(arr):
    """
    The square root of the sum of squares:
    
    .. math::

       \sqrt{\sum_{i}^{N} X_{i}^{2}}

    This can be used to solve the Pythagorean equality, or to calculate
    Euclidian distance.
    """

    # Make sure to treat it as an array:
    arr = np.asarray(arr)
    
    return np.sqrt(np.sum(arr**2))

def nearest_coord(vol, in_coords, tol=10):
    """
    Find the coordinate in vol that contains data (not nan) that is spatially
    closest to in_coord  
    """
    vol_idx = np.where(~np.isnan(vol))

    # Get the Euclidian distance for the in_coord from all the volume
    # coordinates: 
    d_x = in_coords[0] - vol_idx[0]
    d_y = in_coords[1] - vol_idx[1]
    d_z = in_coords[2] - vol_idx[2]
    
    delta = np.sqrt(d_x**2 + d_y**2 + d_z**2)

    min_delta  = np.nanmin(delta)
    
    # If this is within the requested tolerance: 
    if min_delta <= tol:
        idx = np.where(delta == np.min(delta))[0]
        return [vol_idx[i][idx] for i in range(3)]

    # Otherwise, return None: 
    else:
        return None 
    
def coeff_of_determination(data, model, axis=-1):
    """

     http://en.wikipedia.org/wiki/Coefficient_of_determination

              _                                            _
             |    sum of the squared residuals              |
    R^2 =    |1 - ---------------------------------------   | * 100
             |_    sum of the squared mean-subtracted data _|


    """
    # There's no point in doing any of this: 
    if np.all(data==0.0) and np.all(model==0.0):
        return np.nan
    
    residuals = data - model
    ss_err = np.sum(residuals ** 2, axis=axis)

    demeaned_data = data - np.mean(data,-1)[...,np.newaxis]
    ss_tot = np.sum(demeaned_data **2, axis=axis)

    # Don't divide by 0:
    if np.all(ss_tot==0.0):
        return np.nan
    
    return 1 - (ss_err/ss_tot)

def rescale(arr):
    """

   rescale an array into [0,1]

    """
    # Start by moving the minimum to 0:
    min_arr = np.nanmin(arr)

    if min_arr<0:
        arr += np.abs(min_arr)
    else:
        arr -= np.abs(min_arr)
        
    return arr/np.nanmax(arr)


def rmse(arr1, arr2, axis=-1):
    """
    Calculate the root of the mean square error (difference) between two arrays

    Parameters
    ----------

    arr1, arr2: array-like
       Need to have the same shape

    axis: int, optional
       The axis over which the averaging step is done.

    
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if has_numexpr:
      return np.sqrt(np.mean(numexpr.evaluate('(arr1 - arr2) ** 2'), axis=axis))
    else:
      return np.sqrt(np.mean((arr1-arr2)**2, axis=axis))

def seed_corrcoef(seed, target):
    """
    Compute seed-based correlation coefficient

    Parameters
    ----------
    seed: a single 1-d arrays, shape (n,)

    target: many 1-d arrays stacked. shape (m,n)
       These will each be compared to the seed

    Returns
    -------
    The correlation coefficient between the seed and each of the targets
    
    """
    x = target - np.mean(target, -1)[..., np.newaxis]
    y = seed - np.mean(seed)
    xx = np.sum(x ** 2, -1)
    yy = np.sum(y ** 2, -1)
    xy = np.dot(x, y)
    r = xy / np.sqrt(xx * yy)

    return r

        
