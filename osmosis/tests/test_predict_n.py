import numpy as np
import numpy.testing as npt
import nibabel as nib
import os

import osmosis.multi_bvals as sfm_mb
import osmosis.model.sparse_deconvolution as sfm
import osmosis.predict_n as pn
import osmosis.utils as ozu

# Mock b value array to be used in most tests
bvals_t = np.array([5, 5, 10, 2010, 1005, 950, 1950, 1000, 1005,
                    995, 2005,2010, 1995])

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

ad = {1000:1.6386920952169737, 2000:1.2919249903637751, 3000:0.99962593218241236}
rd = {1000:0.33450124887561905, 2000:0.28377379537043729, 3000:0.24611723207420028}

data_path = "/biac4/wandell/data/klchan13/100307/Diffusion/data"
data_pv = nib.load(os.path.join(data_path, "red_data.nii.gz")).get_data()
bvals_pv = np.loadtxt(os.path.join(data_path, "bvals"))
bvecs_pv = np.loadtxt(os.path.join(data_path, "bvecs"))

bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals_pv)
all_b_inds = np.where(rounded_bvals != 0)

mask_orig = nib.load(os.path.join(data_path, 'nodif_brain_mask.nii.gz')).get_data()
mask_pv = np.zeros(mask_orig.shape)
mask_pv[0, 0, 0:2] = 1

actual_all = np.squeeze(data_pv[np.where(mask_pv)][:, all_b_inds])
actual_t = data_pv[np.where(mask_pv)][:, b_inds[3]]
actual_t_demeaned = (data_pv[np.where(mask_pv)][:, b_inds[3]] -
                     np.mean(data_pv[np.where(mask_pv)][:, b_inds[3]],-1)[..., None])

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
    actual_vals, predicted_vals = pn.predict_n(data_pv, bvals_pv, bvecs_pv, mask_pv,
                                                              ad, rd, 20, "all")
    npt.assert_equal(np.mean(predicted_vals[1][predicted_vals[1] >1])>800,1)
    npt.assert_equal(np.mean(actual_vals[1][actual_vals[1] >1])>800,1)
    
    # Let's see if the RMSE is reasonable.
    this_rmse = np.sqrt(np.mean((actual_vals - predicted_vals)**2))
    npt.assert_equal(this_rmse<350, 1)
    
def test_grid_predict():
    actual_vals, predicted_vals = pn.predict_grid(data_pv, bvals_pv, bvecs_pv, mask_pv,
                                                              ad, rd, 10)
    npt.assert_equal(np.mean(predicted_vals[1][predicted_vals[1] >1])>800,1)
    npt.assert_equal(np.mean(actual_vals[1][actual_vals[1] >1])>800,1)
    
    this_rmse = np.sqrt(np.mean((actual_all - predicted_vals)**2))
    npt.assert_equal(this_rmse<350, 1)

def test_bval_predict():
    # Now check to see if the values for individual b values is
    # around a value we want
    actual_vals, predicted_vals = pn.predict_n(data_pv, bvals_pv, bvecs_pv, mask_pv,
                                                            ad, rd, 10, "bvals")
    npt.assert_equal(np.mean(predicted_vals[1][predicted_vals[1] >1])>800,1)
    npt.assert_equal(np.mean(actual_vals[1][actual_vals[1] >1])>800,1)
    
    this_rmse = np.sqrt(np.mean((actual_all - predicted_vals)**2))
    npt.assert_equal(this_rmse<350, 1)
    
def test_kfold_xval_bvals_with_mean():
    actual02, predicted02 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 0, 2,
                                                md = "b_mean", mode = "kfold_xval")
    actual12, predicted12 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 1, 2,
                                                md = "b_mean", mode = "kfold_xval")
    actual22, predicted22 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 2, 2,
                                                md = "b_mean", mode = "kfold_xval")

    rmse02 = np.sqrt(np.mean((actual_t - predicted02)**2))
    rmse12 = np.sqrt(np.mean((actual_t - predicted12)**2))
    rmse22 = np.sqrt(np.mean((actual_t - predicted22)**2))

    npt.assert_equal(rmse02<300, 1)
    npt.assert_equal(rmse12<300, 1)
    npt.assert_equal(rmse22<300, 1)
    
def test_kfold_xval_bvals_no_mean():
    actual02, predicted02 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 0, 2,
                                                md = "no_mean", mode = "kfold_xval")
    actual12, predicted12 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 1, 2,
                                                md = "no_mean", mode = "kfold_xval")
    actual22, predicted22 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 2, 2,
                                                md = "no_mean", mode = "kfold_xval")
    
    rmse02 = np.sqrt(np.mean((actual_t_demeaned - predicted02)**2))
    rmse12 = np.sqrt(np.mean((actual_t_demeaned - predicted12)**2))
    rmse22 = np.sqrt(np.mean((actual_t_demeaned - predicted22)**2))

    npt.assert_equal(rmse02<400, 1)
    npt.assert_equal(rmse12<400, 1)
    npt.assert_equal(rmse22<400, 1)
    
def test_predict_bvals_no_mean():
    actual02, predicted02 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 0, 2,
                                                md = "no_mean")
    actual12, predicted12 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 1, 2,
                                                md = "no_mean")
    actual22, predicted22 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 2, 2,
                                                md = "no_mean")

    rmse02 = np.sqrt(np.mean((actual_t_demeaned - predicted02)**2))
    rmse12 = np.sqrt(np.mean((actual_t_demeaned - predicted12)**2))
    rmse22 = np.sqrt(np.mean((actual_t_demeaned - predicted22)**2))

    npt.assert_equal(rmse02<400, 1)
    npt.assert_equal(rmse12<400, 1)
    npt.assert_equal(rmse22<400, 1)
    
def test_predict_bvals_with_mean():
    actual02, predicted02 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 0, 2,
                                                md = "b_mean")
    actual12, predicted12 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 1, 2,
                                                md = "b_mean")
    actual22, predicted22 = pn.predict_bvals(data_pv, bvals_pv, bvecs_pv,
                                                mask_pv, ad, rd, 2, 2,
                                                md = "b_mean")

    rmse02 = np.sqrt(np.mean((actual_t - predicted02)**2))
    rmse12 = np.sqrt(np.mean((actual_t - predicted12)**2))
    rmse22 = np.sqrt(np.mean((actual_t - predicted22)**2))

    npt.assert_equal(rmse02<300, 1)
    npt.assert_equal(rmse12<300, 1)
    npt.assert_equal(rmse22<300, 1)
    
#def test_predict_RD_AD():
    #rmse_b, rmse_mb, AD_order, RD_order = pn.predict_RD_AD(1.3, 2.0, 0.5, 0.9,
                                                                 #4, 4, data_t,
                                                             #bvals_t, bvecs_t,
                                                                       #mask_t)
    #npt.assert_equal(np.abs(rmse_b[0,1] - rmse_mb[0,1])<100, 1)
    #npt.assert_equal(rmse_mb < 400, 1)