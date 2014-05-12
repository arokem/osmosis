import numpy as np
import numpy.testing as npt

import osmosis.snr as snr

# Mock b value array to be used in all tests
bvals_t = np.array([5, 5, 10, 2010, 1005, 950, 1950, 1000])

bval_list_t = [(np.array([0, 0, 0]))]
bval_list_t.append(np.array([1000, 1000, 1000]))
bval_list_t.append(np.array([2000, 2000]))

bval_ind_t = [np.array([0,1,2]), np.array([4,5,7]), np.array([3,6])]

bvals_scaled_t = np.array([0, 0, 0, 2000, 1000, 1000, 2000, 1000])

unique_b_t = np.array([0,1000,2000])

# Mock data to be used in all tests
#data_t = 2000 + abs(np.random.randn(2,2,2,8)*1000)
data_t = np.zeros([2,2,2,8])
data_t[:,:,:,0:3] = 2000 + abs(np.random.randn(2,2,2,3)*500) #Data for b values 0.005, 0.005, 0.010
data_t[:,:,:,3] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) #Data for b value 2.010
data_t[:,:,:,4:6] = 1000 + abs(np.random.randn(2,2,2,2)*500) #Data for b values 1.005, 0.950
data_t[:,:,:,6] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) #Data for b values 1.950
data_t[:,:,:,7] = np.squeeze(1000 + abs(np.random.randn(2,2,2,1)*500)) #Data for b value 1.000

# Mock mask to be used in all tests
mask_t = np.zeros([2,2,2])
mask_t[:,:,1] = 1

def test_separate_bvals():
    
    bval_list, bval_ind, unique_b, bvals_scaled = snr.separate_bvals(bvals_t)
    npt.assert_equal(unique_b, unique_b_t)
    npt.assert_equal(bvals_scaled, bvals_scaled_t)
    
    for i in np.arange(len(bval_list)):
        npt.assert_equal(np.squeeze(bval_list)[i], bval_list_t[i])
        npt.assert_equal(np.squeeze(bval_ind)[i], bval_ind_t[i])
    
def test_b_snr():
    
    idx_mask_t = (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]))
    test_data = data_t[idx_mask_t]
    
    snr_unbiased_t = snr.calculate_snr(test_data[:,np.array([0,1,2])],test_data[:,np.array([3,6])])
    bsnr_t = np.zeros([2,2,2])
    
    bsnr_t[idx_mask_t] = np.squeeze(snr_unbiased_t)
    bsnr_t[np.where(~np.isfinite(bsnr_t))] = 0
    
    npt.assert_equal(bsnr_t, snr.b_snr(data_t, bvals_t, 2, mask_t))

def test_all_snr():
    disp_snr_t = np.zeros([2,2,2])
    bvals0_ind_t = np.array([0,1,2])
    
    disp_snr_t = snr.iter_snr(data_t, mask_t, disp_snr_t, bvals0_ind_t)
    
    npt.assert_equal(snr.all_snr(data_t, bvals_t, mask_t), disp_snr_t)
