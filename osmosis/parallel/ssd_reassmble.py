# Reassmble the pieces into one file per condition, removing the cruft along
# the way...

import os
import nibabel as ni
import osmosis.utils as ozu
import osmosis.io as oio
import numpy as np

alphas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
l1_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

data_path = '/hsgs/u/arokem/tmp/'

for subject in ['FP']: #['HT']
    subject_path = os.path.join(oio.data_path, subject)
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    wm_nifti = ni.load(wm_mask_file)
    wm_data = wm_nifti.get_data()
    wm_aff = wm_nifti.get_affine()
    n_wm_vox = np.sum(wm_data)
    wm_idx = np.where(wm_data==1)

    for b in [1000, 2000, 4000]:
        ad_rd = oio.get_ad_rd(subject, b)
        for data_i, data in enumerate(oio.get_dwi_data(b, subject)):
            file_stem = (data_path + '%s/'%subject +
                         data[0].split('/')[-1].split('.')[0])
            for l1_ratio in l1_ratios:
                for alpha in alphas:
                    new_vol = ozu.nans(wm_data.shape + (150,))
                    
                    new_fname = "%s_SSD_l1ratio%s_alpha%s.nii.gz"%(file_stem,
                                                               l1_ratio,
                                                               alpha)
                    if not os.path.exists(new_fname):
                        print("Reassembling %s"%new_fname)
                        for i in range(int(n_wm_vox/10000)+2):
                            params_file="%s_SSD_l1ratio%s_alpha%s_%03d.nii.gz"%(
                                file_stem,
                                l1_ratio,
                                alpha,
                                i)
                            low = i*10000
                            # Make sure not to go over the edge of the mask:
                            high = np.min([(i+1)*10000,int(np.sum(wm_data))])
                            this_idx = (wm_idx[0][low:high],
                                        wm_idx[1][low:high],
                                        wm_idx[2][low:high])

                            this_data = ni.load(
                                params_file).get_data()[this_idx]
                            
                            new_vol[this_idx] = this_data
                            #Kill your cruft:
                            os.system('rm %s'%params_file)

                        ni.Nifti1Image(new_vol, wm_aff).to_filename(new_fname)
                    
                    
