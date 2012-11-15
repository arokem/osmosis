import os

import numpy as np
import numpy.testing as npt

import osmosis as oz
from osmosis.model.dti import TensorModel
from osmosis.model.base import SphereModel

import osmosis.model.analysis as ozm

data_path = os.path.split(oz.__file__)[0] + '/data/'

# We'll use these two identical models in a few tests below:
TM1 = TensorModel(data_path + 'small_dwi.nii.gz',
                          data_path + 'dwi.bvecs',
                          data_path + 'dwi.bvals')
    
TM2 = TensorModel(data_path + 'small_dwi.nii.gz',
                          data_path + 'dwi.bvecs',
                          data_path + 'dwi.bvals')
    
def test_overfitting_index():
    """
    Testing the calculation of the overfitting index

    """
    ii = ozm.overfitting_index(TM1, TM2)
    # Since the models are identical, this should all be zeros: 
    npt.assert_equal(ii, np.zeros(ii.shape))


def test_relative_mae():
    """
    Testing the calculation of relative mean absolute error.
    """
    ii = ozm.relative_mae(TM1, TM2)
    # Since the models are identical, this should all be infs: 
    npt.assert_equal(ii, np.ones(ii.shape) * np.inf)

def test_rsquared():
    ii = ozm.rsquared(TM1, TM2)

    # Test for regression:
    npt.assert_almost_equal(ii,
    np.array([[[ 0.41738295,  0.5686638 ,  0.66632678,  0.66796424],
        [ 0.55782746,  0.52997752,  0.65248008,  0.79295422]],

       [[ 0.49519897,  0.52195252,  0.70362685,  0.62545745],
        [ 0.43410031,  0.56910023,  0.76395852,  0.73071651]],

       [[ 0.50371373,  0.56810418,  0.53169063,  0.60985997],
        [ 0.53667339,  0.69261167,  0.70018453,  0.63229423]]]))


def test_noise_ceiling():
    """
    Test the calculation of the noise ceiling
    
    """

    out_coeffs, out_lb, out_ub = ozm.noise_ceiling(TM1, TM2)

    npt.assert_almost_equal(out_coeffs, np.ones(out_coeffs.shape))
    npt.assert_almost_equal(out_lb, np.ones(out_lb.shape))
    npt.assert_almost_equal(out_ub, np.ones(out_ub.shape))


def test_coefficient_of_determination():
    """
    Test the computation of coefficient of determination
    """
    cod = ozm.coeff_of_determination(TM1, TM2)

    # Test for regressions:
    npt.assert_almost_equal(cod,
    np.array([[[ -3.44661890e+00,  -9.56966221e-01,  -2.95984527e-01,
          -2.14809778e-01],
        [ -1.13823857e+00,  -1.47624354e+00,  -2.49836988e-01,
           4.45058522e-01]],

       [[ -2.12348903e+00,  -1.03695127e+00,   3.43880861e-03,
          -5.08955429e-01],
        [ -2.70970026e+00,  -8.62731412e-01,   3.21255708e-01,
           1.52058544e-01]],

       [[ -1.54499435e+00,  -1.12129147e+00,  -1.34573166e+00,
          -5.70547139e-01],
        [ -1.31661328e+00,  -1.31546355e-02,  -9.43582307e-03,
          -4.12026495e-01]]]))

def test_pdd_reliability():
    """
    Test the calculation of model reliability by PDD
    """
    reliab = ozm.pdd_reliability(TM1, TM2)
    npt.assert_equal(reliab, np.zeros(reliab.shape))
    

def test_model_params_reliability():
    """
    Test the calculation of model params reliability by vector angle:
    """
    reliab = ozm.model_params_reliability(TM1, TM2)
    npt.assert_equal(reliab, np.zeros(reliab.shape))

        
def test_relative_rmse():
    """
    Test the calculation of relative RMSE from two model objects

    While you're at it, test the SphereModel class as well.
    
    """
    Model1 = SphereModel(data_path+'small_dwi.nii.gz',
                             data_path + 'dwi.bvecs',
                             data_path + 'dwi.bvals',)

    Model2 = SphereModel(data_path+'small_dwi.nii.gz',
                             data_path + 'dwi.bvecs',
                             data_path + 'dwi.bvals',)

    # Since we have exactly the same data in both models, the rmse between them
    # is going to be 0 everywhere, which means that the relative rmse is
    # infinite... 
    npt.assert_equal(ozm.relative_rmse(Model1, Model2),
                     np.inf * np.ones(Model1.shape[:-1]))
    
