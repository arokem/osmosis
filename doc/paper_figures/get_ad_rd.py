# Get AD and RD for the CC
import os
import numpy as np

import nibabel as ni

import osmosis
import osmosis.utils as ozu
import osmosis.model.dti as dti
import osmosis.io as oio

top_n = 250
data_path = '/home/arokem/data/osmosis/'

for subject in ['FP']:#, 'HT']:
    subject_path = os.path.join(data_path, subject)
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    wm_nifti = ni.load(wm_mask_file)
    wm_data = wm_nifti.get_data()

    cc_file = os.path.join(data_path,subject,'%s_cc.nii.gz'%subject)
    cc_nifti = ni.load(cc_file)
    cc_data = cc_nifti.get_data()

    idx = np.where(np.logical_and(cc_data, wm_data))

    for b in [1000, 2000, 4000]:
        data_1, data_2 = oio.get_dwi_data(b, subject=subject)
        TM1 = dti.TensorModel(*data_1, mask=wm_data)
        TM2 = dti.TensorModel(*data_2, mask=wm_data)
        for TM in [TM1, TM2]:
            AD = TM.axial_diffusivity[idx]
            RD = TM.radial_diffusivity[idx]
            MD = TM.mean_diffusivity[idx]
            print('For: %s and b=%s: AD=%1.4f, RD=%1.4f, MD=%1.4f'%(
                   subject, b, np.median(AD), np.median(RD), np.median(MD)))
            idx_AD = np.argmin(np.abs(AD - np.median(AD)))
            idx_RD = np.argmin(np.abs(RD - np.median(RD)))
            idx_MD = np.argmin(np.abs(MD - np.median(MD)))
            print('For: %s and b=%s: AD idx=%s, RD idx =%s, MD idx=%s'%(
                   subject, b, idx_AD, idx_RD, idx_MD))
            
            
            
    
    
