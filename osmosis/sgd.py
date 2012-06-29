"""

Stochastic Gradient Descent

This is adapted from Kendrick Kay's Matlab SGD code.

"""

import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

# In accordance with the conditions of re-distribution of code written by KNK
# the following comment appears here:
# I Love MATLAB(tm) 

# OK - now when we have that out of the way, on to bigger and better

def stochastic_gradient_descent(y, X, momentum=0,
                                prop_select=0.01,
                                step_size=0.1,
                                non_neg=True,
                                prop_bad_checks=0.1,
                                check_error_iter=10,
                                max_error_checks=10,
                                converge_on_r=0.2,
                                verbose=True,
                                plot=True):
    """

    Solve y=Xh for h, using a stochastic gradient descent.
    
    Parameters
    ----------

    y: 1-d array of shape (N)
        The data
        
    X: ndarray of regressors. May be either sparse or dense. of shape (N, M)
       The regressors
    
    prop_select: float (0-1, default 0.01)
        What proportion of the samples to evaluate in each iteration of the
        algorithm.

    step_size: float, optional (default: 0.05). 
        The increment of parameter update in each iteration
                
    non_neg: Boolean, optional (default: True)
        Whether to enforce non-negativity of the solution.

    prop_bad_checks: float (default: 0.1)
       If this proportion of error checks so far has not yielded an improvement
       in r squared, we halt the optimization.
       
    check_error_iter: int (default:10)
        How many rounds to run between error evaluation for
        convergence-checking.

    max_error_checks: int (default: 10)
        Don't check errors more than this number of times if no improvement in
        r-squared is seen.
       
    converge_on_r: float (default: 1)
      a percentage improvement in rsquared that is required each time to say
      that things are still going well.

    verbose: Boolean (default: True).
       Whether to display information in each iteration

    plot: whether to generate a plot of the progression of the optimization
    
    Returns
    -------
    h_best: The best estimate of the parameters.
    
    
    """

    num_data = y.shape[0]
    num_regressors = X.shape[1]
    n_select = np.round(prop_select * num_data)

    # Initialize the parameters at the origin:
    h = np.zeros(num_regressors)

    # If nothing good happens, we'll return that in the end:
    h_best = np.zeros(num_regressors)
    
    gradient = np.zeros(num_regressors)
    
    iteration = 1
    count = 1
    ss_residuals = []  # This will hold the residuals in each iteration
    ss_residuals_min = np.inf  # This will keep track of the best solution so far
    ss_residuals_to_mean = np.sum((y - np.mean(y))**2) # The variance of y
    rsq_max = -np.inf   # This will keep track of the best r squared so far
    count_bad = 0  # Number of times estimation error has gone up.
    error_checks = 0  # How many error checks have we done so far

        
    while 1:
        # indices of data points to select
        idx = np.floor(np.random.rand(n_select) * num_data).astype(int);

        # Select for this round 
        y0 = y[idx]
        X0 = X[idx]

        if iteration>1: 
            # The sum of squared error given the current parameter setting: 
            sse = np.sum((y - spdot(X,h))**2)
            # The gradient is (Kay 2008 supplemental page 27): 
            gradient = (spdot(X0.T, spdot(X0,h) - y0)) + momentum*gradient
            # Normalize to unit-length
            unit_length_gradient = gradient / np.sqrt(np.dot(gradient, gradient))
            # Update the parameters in the direction of the gradient:
            h -= step_size * unit_length_gradient

            if non_neg:
                # Set negative values to 0:
                h[h<0] = 0

        # Every once in a while check whether it's converged:
        if np.mod(iteration, check_error_iter):
            # This calculates the sum of squared residuals at this point:
            ss_residuals.append(np.sum(np.power(y - spdot(X,h), 2)))
            rsq_est = rsq(ss_residuals[-1], ss_residuals_to_mean)
            if verbose:
                print("Itn #:%03d | SSE: %.1f | R2=%.1f "%
                      (iteration,
                       ss_residuals[-1],
                          rsq_est))

            # Did we do better this time around? 
            if  ss_residuals[-1]<ss_residuals_min:
                # Update your expectations about the minimum error:
                ss_residuals_min = ss_residuals[-1]
                n_iterations = iteration # This holds the number of iterations
                                        # for the best solution so far.
                h_best = h # This holds the best params we have so far

                # Are we generally (over iterations) converging on
                # improvement in r-squared?
                if rsq_est>rsq_max*(1+converge_on_r/100):
                    rsq_max = rsq_est
                    count_bad = 0 # We're doing good. Null this count for now
                else:
                    count_bad += 1
            else:
                count_bad += 1
                
            if count_bad >= np.max([max_error_checks,
                                    np.round(prop_bad_checks*error_checks)]):
                print("\nOptimization terminated after %s iterations"%iteration)
                print("R2= %.1f "%rsq_max)
                print("Sum of squared residuals= %.1f"%ss_residuals_min)

                if plot:
                    fig, ax = plt.subplots()
                    ax.plot(ss_residuals)
                    ax.set_xlabel("Iteration #")
                    ax.set_ylabel(r"$\sum{(\hat{y} - y)^2}$")
                # Break out here, because this means that not enough
                # improvement has happened recently
                return h_best
            error_checks += 1
        iteration += 1


        
def rsq(ss_residuals,ss_residuals_to_mean):
    """
    Helper function which calculates $R^2 = \frac{1-SSE}{\sigma^2}$
    """

    return 100 * (1 - ss_residuals/ss_residuals_to_mean)

    
def spdot(A, B):
  """The same as np.dot(A, B), except it works even if A or B or both
  might be sparse.

  See discussion here:
  http://mail.scipy.org/pipermail/scipy-user/2010-November/027700.html
  """
  if sps.issparse(A) and sps.issparse(B):
      return A * B
  elif sps.issparse(A) and not sps.issparse(B):
      return (A * B).view(type=B.__class__)
  elif not sps.issparse(A) and sps.issparse(B):
      return (B.T * A.T).T.view(type=A.__class__)
  else:
      return np.dot(A, B)
