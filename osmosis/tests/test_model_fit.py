import numpy as np
import osmosis.model.dti as dti
from osmosis.utils import ols_matrix

import osmosis.model_fit as mf

import numpy.testing as npt

saved_file = 'no'
# Mock b value array to be used in all tests
bvals_t = np.array([0.005, 0.005, 0.010, 2.010, 1.005, 0.950, 1.950, 1.000])

bval_list_t = [(np.array([0.005, 0.005, 0.010]))]
bval_list_t.append(np.array([1.005, 0.950, 1.000]))
bval_list_t.append(np.array([2.010, 1.950]))

bval_ind_t = [np.array([0,1,2]), np.array([4,5,7]), np.array([3,6])]

unique_b_t = np.array([0,1,2])

# Mock b vector data to be used in all tests
bvecs_t = np.zeros([3,8])
bvecs_t[0,:] = np.array([1,0,0,-1,0,0,1/np.sqrt(3),-1/np.sqrt(3)])
bvecs_t[1,:] = np.array([0,1,0,0,-1,0,1/np.sqrt(3),-1/np.sqrt(3)])
bvecs_t[2,:] = np.array([0,0,1,0,0,-1,1/np.sqrt(3),-1/np.sqrt(3)])

# Mock data to be used in all tests
data_t = np.zeros([2,2,2,8])
data_t[:,:,:,0:3] = 2000 + abs(np.random.randn(2,2,2,3)*500) #Data for b values 0.005, 0.005, 0.010
data_t[:,:,:,3] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) #Data for b value 2.010
data_t[:,:,:,4:6] = 1000 + abs(np.random.randn(2,2,2,2)*500) #Data for b values 1.005, 0.950
data_t[:,:,:,6] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) #Data for b values 1.950
data_t[:,:,:,7] = np.squeeze(1000 + abs(np.random.randn(2,2,2,1)*500)) #Data for b value 1.000

# Mock mask to be used in all tests
mask_t = np.zeros([2,2,2])
mask_t[:,:,1] = 1

# Mock index arrays to be used in all tests
idx_array = np.array([0,1,2])
idx_mask_t = np.where(mask_t)

def test_include_b0vals():
    bvals_wb0_t = [np.array([0.005, 0.005, 0.010, 1.005, 0.950, 1.000])]
    bvals_wb0_t.append(np.array([0.005, 0.005, 0.010, 2.010, 1.950]))
    
    bval_ind_wb0_t = [np.array([0,1,2,4,5,7]), np.array([0,1,2,3,6])]
    bval_ind_wb0_a, bvals_wb0_a = mf.include_b0vals(idx_array, bval_ind_t, bval_list_t)
    
    npt.assert_equal(bvals_wb0_t, bvals_wb0_a)
    npt.assert_equal(bval_ind_wb0_t, bval_ind_wb0_a)
    
    return bvals_wb0_t, bval_ind_wb0_t
    
def test_log_prop_vals():
    bvals_wb0_t, bval_ind_wb0_t = test_include_b0vals()
    
    tensor_prop1 = dti.TensorModel(data_t[:,:,:,bval_ind_wb0_t[0]], bvecs_t[:,bval_ind_wb0_t[0]], bvals_wb0_t[0], mask = mask_t, params_file = 'temp')
    tensor_prop2 = dti.TensorModel(data_t[:,:,:,bval_ind_wb0_t[1]], bvecs_t[:,bval_ind_wb0_t[1]], bvals_wb0_t[1], mask = mask_t, params_file = 'temp')
    
    log_prop_t = [np.log(tensor_prop1.fractional_anisotropy[idx_mask_t]+0.01), np.log(tensor_prop2.fractional_anisotropy[idx_mask_t]+0.01)]
    log_prop_a = mf.log_prop_vals('FA', saved_file, data_t, bvecs_t, idx_mask_t, idx_array, bval_ind_wb0_t, bvals_wb0_t, mask_t)
    
    npt.assert_equal(log_prop_t[0], log_prop_a[0])
    npt.assert_equal(log_prop_t[1], log_prop_a[1])
    
    return log_prop_t
    
def test_ls_fit_b():
    log_prop_t = test_log_prop_vals()
    b_matrix_t = np.matrix([[1,2], [1,1]]).T
    b_inv = ols_matrix(b_matrix_t)
    ls_fit_FA_t = np.dot(b_inv, np.matrix(log_prop_t))
    
    npt.assert_equal(ls_fit_FA_t, mf.ls_fit_b(log_prop_t, unique_b_t))
    
    return ls_fit_FA_t
    
def test_slope():
    ls_fit_FA_t = test_ls_fit_b()
    # Test for FA
    slopeProp_all_t = np.zeros([2,2,2])
    slopeProp_all_t[idx_mask_t] = np.squeeze(np.array(ls_fit_FA_t[0,:][np.isfinite(ls_fit_FA_t[0,:])]))
    
    npt.assert_equal(slopeProp_all_t, mf.slope(data_t, bvals_t, bvecs_t, 'FA', mask = mask_t, saved_file = 'no'))