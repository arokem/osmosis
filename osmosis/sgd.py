"""

Stochastic Gradient Descent

This is adapted from Kendrick Kay's Matlab SGD code.

"""


# The conditions of re-distribution of code written by KNK require that the
# following comment appear here: 
# I <3 matlab

# OK - now when we have that out of the way, on to bigger and better

import numpy as np

def stochastic_gradient_descent(y, X, prop_select=0.1, final_step=0.05,
                                non_neg=True):
    """
    Solve y=Xh for h, using a stochastic gradient descent approach.
    
    Parameters
    ----------

    y: 1-d array of shape (N)
        The data
      
    X: ndarray of regressors. May be either sparse or dense. of shape (N, M)
       The regressors
    
    prop_select: float (0-1)
        What proportion of the samples to evaluate in each iteration of the
        algorithm.

    final_step: float, optional (default: 0.05). 
        The increment of parameter update in each iteration
                
    non_neg: Boolean, optional (default: True)
        Whether to enforce non-negativity of the solution.

    check_error_iter: int
        How many rounds to run between error evaluation for
        convergence-checking. 
       
    
    
    Returns
    -------

    
    """

    
    num_data = y.shape[0]
    num_regressors = X.shape[1]

    n_select = np.round(p_select * num_data)

    iteration = 1
    count = 1
    while 1: 
        
        # indices of data points to select
        idx = np.ceil(np.random.rand(N)*p);

        # Select for this round 
        y0 = y[idx]
        X0 = X[idx]

        if iteration>1: 
            # The sum of squared error given the current parameter setting: 
            sse = np.sum((y - np.dot(X,h))**2)

            # The gradient is: 
            gradient = -((y0 - np.dot(np.dot(X0,h)).T , X0)).T;
            # Normalize
            unit_length_gradient = gradient / np.sqrt(np.dot(gradient, gradient))

            # Update the parameters:
            h -= np.dot(final_step, gradient)

            if non_neg:
                # Set negative values to 0:
                h[h<0] = 0

        # Every once in a while check whether it's converged:
        if np.mod(iteration, prop_check_error):
            # This calculates the sum of squared residuals at this point:
            ss_residuals = np.sum(np.power(y - np.dot(X,h), 2))
            
