# Make the white matter masks
import os
import numpy as np

import nibabel as ni
import osmosis.wm_mask as wm

data_path = '/Users/arokem/projects/data_osmosis_raw/'
rrmse_dti = {}
rrmse_ssd = {}
for subject in ['FP', 'HT']:
    subject_path = os.path.join(data_path, subject)
    # Start by making the white-matter mask, if it doesn't exist yet:
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    if ~os.path.exists(wm_mask_file):
        seg_file = os.path.join(subject_path, '%s_t1_class.nii.gz'%subject)
        # We use the dti data from the b=2k to generate the WM mask:
        listdir = os.listdir(subject_path)
        find_1k = []
        for this_file in listdir:
            if this_file.find('2000')>0:
                find_1k.append(this_file)
        # Use the first one:
        the_files = find_1k[0].split('.')[0]
        wm.make_wm_mask(seg_file, os.path.join(subject_path, the_files),
                        wm_mask_file)

