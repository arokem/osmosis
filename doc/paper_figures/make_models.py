import os
import numpy as np

import nibabel as ni

import osmosis
import osmosis.utils as ozu
import osmosis.model.dti as dti
import osmosis.model.sparse_deconvolution as ssd
import osmosis.model.analysis as ozm
import osmosis.io as oio

data_path = oio.data_path

rrmse_dti = {}
rrmse_ssd = {}
for subject in ['FP', 'HT']:
    subject_path = os.path.join(data_path, subject)
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    wm_nifti = ni.load(wm_mask_file).get_data()
    wm_mask = np.zeros(wm_nifti.shape)
    wm_mask[np.where(wm_nifti==1)] = 1
    wm_idx = np.where(wm_nifti>0)
    rrmse_dti[subject] = {}
    rrmse_ssd[subject] = {}
    for b in [1000, 2000, 4000]:
        data_1, data_2 = oio.get_dwi_data(b, subject=subject)
        TM1 = dti.TensorModel(*data_1, mask=wm_mask)
        TM2 = dti.TensorModel(*data_2, mask=wm_mask)
        rrmse_dti[subject][b] = ozm.cross_predict(TM1, TM2)
        print subject
        print b
        rmse_mask = rrmse_dti[subject][b][wm_idx]
        print "DTI: %s voxels above 1"%(len(np.where(rmse_mask>1)[0])/float(len(rmse_mask)))
        # 
        ad_rd = oio.get_ad_rd(subject, b)
        SD1 = ssd.SparseDeconvolutionModel(*data_1, mask=wm_mask,
            axial_diffusivity=ad_rd[0]['AD'],
            radial_diffusivity=ad_rd[0]['RD'])
        SD2 = ssd.SparseDeconvolutionModel(*data_2, mask=wm_mask,
            axial_diffusivity=ad_rd[1]['AD'],
            radial_diffusivity=ad_rd[1]['RD'])
        rrmse_ssd[subject][b] = ozm.cross_predict(SD1, SD2)
        print "SSD: %s voxels above 1"%(len(np.where(rmse_mask>1)[0])/float(len(rmse_mask)))
        rmse_mask = rrmse_ssd[subject][b][wm_idx]

        
