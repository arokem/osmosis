import os
import numpy as np
import nibabel as nib

import osmosis.multi_bvals as sfm_mb
import osmosis.model.sparse_deconvolution as sfm
import osmosis.predict_n as pn
import osmosis.snr as snr

data_path_dwi = "/biac4/wandell/data/klchan13/100307/Diffusion/data"
red_data = nib.load(os.path.join(data_path_dwi, "data.nii.gz")).get_data()
mask_data = nib.load(os.path.join(data_path_dwi, "nodif_brain_mask.nii.gz")).get_data()
bvals = np.loadtxt(os.path.join(data_path_dwi,'bvals'))
bvecs = np.loadtxt(os.path.join(data_path_dwi,'bvecs'))

mask = np.zeros(mask_data.shape)
mask[0:2, 0:2, 0:2] = 1

ad = {1000:1, 2000:1, 3000:1}
rd = {1000:0, 2000:0, 3000:0}
bval_list, b_inds, unique_b, rounded_bvals = snr.separate_bvals(bvals)
                     
mb = sfm_mb.SparseDeconvolutionModelMultiB(red_data, bvecs, bvals,
                                                      mask = mask,
                                           axial_diffusivity = ad,
                                          radial_diffusivity = rd,
                                             params_file = "temp")
sd = sfm.SparseDeconvolutionModel(red_data, bvecs, bvals, mask = mask,
                                               axial_diffusivity = ad,
                                              radial_diffusivity = rd,
                                                 params_file = "temp")

def bare_predict():
    mod.predict(bvecs[:, b_inds[1][0:9]], bvals[b_inds[1][0:9]])
    
def predict_many():
    pn.predict_n(red_data, bvals, bvecs, mask, ad, rd, 10, "all", over_sample = 90)

def func():
    #mod.regressors()
    #mod._calc_rotations(0, bvals[b_inds[1][0:9]]/1000, bvecs[:, b_inds[1][0:9]])
    bare_predict()