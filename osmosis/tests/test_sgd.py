import numpy as np
import numpy.testing as npt

import scipy.sparse as sps

import osmosis.sgd
reload(osmosis.sgd)
from osmosis.sgd import stochastic_gradient_descent as sgd


def test_sgd():

    # Set up the regression:
    beta = np.random.rand(10)
    X = np.random.randn(1000,10)

    y = np.dot(X, beta)
    beta_hat = sgd(y,X, plot=False)

    beta_hat_sparse = sgd(y, sps.csr_matrix(X), plot=False)

    # We should be able to get back the right answer for this simple case
    npt.assert_array_almost_equal(beta, beta_hat, decimal=1)
    npt.assert_array_almost_equal(beta, beta_hat_sparse, decimal=1)

    
if __name__=="__main__":
     test_sgd()
