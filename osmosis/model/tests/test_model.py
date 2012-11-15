import os
import tempfile
import warnings

import numpy as np
import numpy.testing as npt
import scipy.stats as stats

import nibabel as ni

import osmosis as oz
import osmosis.model as ozm
import osmosis.fibers as ozf
import osmosis.tensor as ozt
import osmosis.io as mio
import osmosis.utils as ozu

# Initially, we want to check whether the data is available (would have to be
# downloaded separately, because it's huge): 
data_path = os.path.split(oz.__file__)[0] + '/data/'
if 'dwi.nii.gz' in os.listdir(data_path):
    no_data = False
else:
    no_data = True

    



        
def test_relative_rmse():
    """
    Test the calculation of relative RMSE from two model objects

    While you're at it, test the SphereModel class as well.
    
    """
    Model1 = ozm.SphereModel(data_path+'small_dwi.nii.gz',
                             data_path + 'dwi.bvecs',
                             data_path + 'dwi.bvals',)

    Model2 = ozm.SphereModel(data_path+'small_dwi.nii.gz',
                             data_path + 'dwi.bvecs',
                             data_path + 'dwi.bvals',)

    # Since we have exactly the same data in both models, the rmse between them
    # is going to be 0 everywhere, which means that the relative rmse is
    # infinite... 
    npt.assert_equal(ozm.relative_rmse(Model1, Model2),
                     np.inf * np.ones(Model1.shape[:-1]))
    
                            
