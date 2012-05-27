import numpy as np
import matplotlib.pyplot as plt

import os
import nibabel as ni 
import osmosis as oz
import osmosis.model as ozm
import osmosis.viz as viz
import osmosis.utils as ozu
reload(ozm)
reload(ozu)

data_path = '/home/arokem/projects/osmosis/osmosis/data/'

file_roots = ['0011_01_DWI_2mm150dir_2x_b1000_aligned_trilin',
              '0007_01_DTI_2mm_150dir_2x_b2000_aligned_trilin',
              '0009_01_DWI_2mm150dir_2x_b1000_aligned_trilin']

all_bvals = [1000, 2000, 4000]
AD = [1.7139, 1.4291, 0.8403]
RD = [0.3887, 0.3507, 0.2369 ]


for b_idx, file_root in enumerate(file_roots): 
    
    dwi, bvecs, bvals = [data_path + file_root + ext for ext in['.nii.gz',
                                                                '.bvecs',
                                                                '.bvals']]
    
    print ("Working on %s"%dwi)
    brain_mask = ni.load(data_path + 'brainMask.nii.gz').get_data()

    brain_idx = np.array(np.where(brain_mask))

    vox_per_chunk = 1000
    n_chunks = int(np.round(np.sum(brain_mask)))/vox_per_chunk
                                     
    for chunk in range(n_chunks):
        print "Working on chunk %s"%chunk
        if vox_per_chunk*chunk+1 > brain_idx.shape[-1]:
            this_idx = brain_idx[:,vox_per_chunk * chunk:]
        else:
            this_idx = brain_idx[:,vox_per_chunk * chunk:vox_per_chunk*(chunk+1)]
        mask_array = np.zeros((81, 106, 76))
        mask_array[(this_idx[0], this_idx[1], this_idx[2])] = 1

        Model = ozm.MultiCanonicalTensorModel(dwi,
                                          bvecs,
                                          bvals,
                                          mask=mask_array,
                                          params_file = '/home/arokem/data/ModelFits/MultiCanonicalTensor2/MultiCanonicalTensorb%s_%03d.nii.gz'%(all_bvals[b_idx], chunk+1),
                                          # These are based on the calculations
                                          # in GetADandRD_multi_bval:
                                          radial_diffusivity=RD[b_idx],
                                          axial_diffusivity=AD[b_idx]) 
        
        # Force a save of the params to file: 
        Model.model_params
    

for b_idx, file_root in enumerate(file_roots): 
    

    vol = np.ones((81,106,76,4)) * np.nan

    for chunk in range(n_chunks):
        if vox_per_chunk*chunk+1 > brain_idx.shape[-1]:
            this_idx = brain_idx[:,vox_per_chunk * chunk:]
        else:
            this_idx = brain_idx[:,vox_per_chunk * chunk:vox_per_chunk*(chunk+1)]
        mask_array = np.zeros((81, 106, 76))
        mask_array[(this_idx[0], this_idx[1], this_idx[2])] = 1
        params_file = '/home/arokem/data/ModelFits/MultiCanonicalTensor2/MultiCanonicalTensorb%s_%03d.nii.gz'%(all_bvals[b_idx], chunk+1)
        this_params = ni.load(params_file).get_data()
        mask_idx = np.where(mask_array)
        vol[mask_idx] = this_params[mask_idx]
    
    fig = viz.mosaic(vol[:,:,:,-1])
    fig.set_size_inches([20,18])

    ni.Nifti1Image(vol, ni.load(params_file).get_affine()).to_filename(
                    '/home/arokem/data/ModelFits/MultiCanonicalTensor/MultiCanonicalTensorCombinedb%s.nii.gz'%all_bvals[b_idx])
    
    
    
    
    
    
