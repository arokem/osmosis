import numpy as np
import numpy.testing as npt
import nibabel as nib
import os

import osmosis.multi_bvals as sfm_mb
import osmosis.model.sparse_deconvolution as sfm
import osmosis.predict_n as pn

# Mock b value array to be used in most tests
bvals_t = np.array([0.005, 0.005, 0.010, 2.010, 1.005, 0.950, 1.950, 1.000, 1.005,
                    0.995, 2.005,2.010, 1.995])

bval_list_t = [(np.array([0, 0, 0]))]
bval_list_t.append(np.array([1000, 1000, 1000, 1000, 1000]))
bval_list_t.append(np.array([2000, 2000, 2000, 2000, 2000]))

bval_ind_t = [np.array([0,1,2]), np.array([4,5,7,8,9]), np.array([3,6,10,11,12])]
bval_ind_rm0_t = [np.array([1,2,4,5,6]), np.array([0,3,7,8,9])]

bvals_scaled_t = np.array([0, 0, 0, 2000, 1000, 1000, 2000, 1000, 1000, 1000,
                           2000, 2000, 2000])

unique_b_t = np.array([0,1,2])

bvecs_t = np.zeros([3,13])
bvecs_t[0,:] = np.array([1,0,0,-1,0,0, 1/np.sqrt(3), -1/np.sqrt(3), 3/np.sqrt(14),
                         1/np.sqrt(26), 2/np.sqrt(12), -2/np.sqrt(12),
                         3/np.sqrt(27)])
bvecs_t[1,:] = np.array([0,1,0,0,-1,0, 1/np.sqrt(3), -1/np.sqrt(3), 2/np.sqrt(14),
                        -3/np.sqrt(26), 2/np.sqrt(12), -2/np.sqrt(12),
                         3/np.sqrt(27)])
bvecs_t[2,:] = np.array([0,0,1,0,0,-1, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(14),
                         4/np.sqrt(26), 2/np.sqrt(12), -2/np.sqrt(12),
                         3/np.sqrt(27)])

# Mock data to be used in most tests
data_t = np.zeros([2,2,2,13])
data_t[:,:,:,0:3] = 2000 + abs(np.random.randn(2,2,2,3)*500)# For b=0
data_t[:,:,:,3] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) # For b=2
data_t[:,:,:,4:6] = 1000 + abs(np.random.randn(2,2,2,2)*500) # For b=1
data_t[:,:,:,6] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) # For b=2
data_t[:,:,:,7:10] = np.squeeze(1000 + abs(np.random.randn(2,2,2,3)*500))#For b=1
data_t[:,:,:,10:13] = np.squeeze(500 + abs(np.random.randn(2,2,2,3)*200))#For b=2

# Mock mask to be used in most tests
mask_t = np.zeros([2,2,2])
mask_t[:,:,1] = 1

ad = {1000:1.5, 2000:1.5, 3000:1.5}
rd = {1000:0.5, 2000:0.5, 3000:0.5}

def test_regressors():
    full_mod_t = sfm_mb.SparseDeconvolutionModelMultiB(data_t, bvecs_t, bvals_t,
                                                                  mask = mask_t,
                                                         axial_diffusivity = ad,
                                                        radial_diffusivity = rd,
                                                           params_file = "temp")
    vec_pool_t = np.arange(len(bval_ind_t[1]))
    np.random.shuffle(vec_pool_t)
    vec_pool_inds = vec_pool_t[0:3]
    vec_combo_t = bval_ind_t[1][vec_pool_inds]
    idx = list(bval_ind_rm0_t[0])

    these_inc0 = list(np.arange(len(bvals_scaled_t)))
    for choice_idx in vec_pool_inds:
        these_inc0.remove(bval_ind_t[1][choice_idx])
        idx.remove(bval_ind_rm0_t[0][choice_idx])

    idx = np.concatenate((idx, bval_ind_rm0_t[1]),0)
    si_t = sorted(idx)
    these_inc0 = np.array(these_inc0)

    these_bvecs_t = bvecs_t[:, these_inc0]
    these_bvals_t = bvals_t[these_inc0]
    this_data_t = data_t[:, :, :, these_inc0]

    mod = pn.sfm_mb.SparseDeconvolutionModelMultiB(this_data_t, these_bvecs_t,
                                                 these_bvals_t, mask = mask_t,
                                                       axial_diffusivity = ad,
                                                      radial_diffusivity = rd,
                                                         params_file = "temp")

    these_regressors_t = [full_mod_t.regressors[0][:, si_t],
                        full_mod_t.regressors[1][:, si_t][si_t, :]]
                        
    # Check to see if the regressors indexed from the full model equal the
    # regressors from the reduced model.
    npt.assert_equal(mod.regressors[0], these_regressors_t[0])
    npt.assert_equal(mod.regressors[1], these_regressors_t[1])
    
def test_all_predict():
    # Now check to see if the values for multi b values is around a
    # value we want
    actual_vals, predicted_vals = pn.predict_n(data_t, bvals_t, bvecs_t, mask_t,
                                                              ad, rd, 20, "all")
    npt.assert_equal(np.mean(predicted_vals[1][predicted_vals[1] >1])>800,1)
    npt.assert_equal(np.mean(actual_vals[1][actual_vals[1] >1])>800,1)

def test_bval_predict():
    # Now check to see if the values for individual b values is
    # around a value we want
    actual_vals, predicted_vals = pn.predict_n(data_t, bvals_t, bvecs_t, mask_t,
                                                            ad, rd, 20, "bvals")
    npt.assert_equal(np.mean(predicted_vals[1][predicted_vals[1] >1])>800,1)
    npt.assert_equal(np.mean(actual_vals[1][actual_vals[1] >1])>800,1)
    
def test_predict_bvals():
    data_path = "/biac4/wandell/data/klchan13/100307/Diffusion/data"
    data_pv = nib.load(os.path.join(data_path, "red_data.nii.gz")).get_data()
    bvals_pv = np.loadtxt(os.path.join(data_path, "bvals"))
    bvecs_pv = np.loadtxt(os.path.join(data_path, "bvecs"))
    mask_orig = nib.load(os.path.join(data_path, 'nodif_brain_mask.nii.gz')).get_data()
    mask_pv = np.zeros(mask_orig.shape)
    mask_pv[0, 0, 0:2] = 1

    actual02, predicted02 = pn.predict_bvals(data_pv, bvals_pv/1000, bvecs_pv,
                                                        mask_pv, ad, rd, 0, 2)
    actual12, predicted12 = pn.predict_bvals(data_pv, bvals_pv/1000, bvecs_pv,
                                                        mask_pv, ad, rd, 1, 2)
    actual22, predicted22 = pn.predict_bvals(data_pv, bvals_pv/1000, bvecs_pv,
                                                        mask_pv, ad, rd, 2, 2)

    rmse02 = np.sqrt(np.mean((actual02 - predicted02)**2))
    rmse12 = np.sqrt(np.mean((actual12 - predicted12)**2))
    rmse22 = np.sqrt(np.mean((actual22 - predicted22)**2))

    npt.assert_equal(rmse02>rmse12, 1)
    npt.assert_equal(rmse12>rmse22, 1)
    
#def test_predict_RD_AD():
    #rmse_b, rmse_mb, AD_order, RD_order = pn.predict_RD_AD(1.3, 2.0, 0.5, 0.9,
                                                                 #4, 4, data_t,
                                                             #bvals_t, bvecs_t,
                                                                       #mask_t)
    #npt.assert_equal(np.abs(rmse_b[0,1] - rmse_mb[0,1])<100, 1)
    #npt.assert_equal(rmse_mb < 400, 1)