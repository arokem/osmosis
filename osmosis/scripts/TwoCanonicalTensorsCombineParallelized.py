"""

Script for fitting the MultiCanonicalTensorModel in parallel over several
files/b-values.

""" 
import numpy as np

import gc
import os
import nibabel as ni 
import osmosis as oz
import osmosis.model as ozm
import osmosis.viz as viz
import osmosis.utils as ozu

from IPython import parallel as p

rc = p.Client()
rc[:].execute('import gc')
rc[:].execute('import numpy as np')
rc[:].execute('import osmosis.model as ozm')
dview = rc[:]
n_engines = len(rc)
print("Running on %s engines"%n_engines)

def para_func(Model, chunk): 
    Model.model_params
    del Model
    gc.collect()
    return "Done with chunk %s:"%chunk

def caller_func(Models, chunks, dview):
    return dview.map_async(para_func, Models, chunks).get()
    
def make_data_set(root):
    """
    Create the full paths to the data given a root
    """ 
    exts = ['.nii.gz', '.bvecs', '.bvals']
    dwi, bvecs, bvals = [data_path + root + ext for ext in exts]
    return dwi, bvecs, bvals 

data_path = os.path.split(oz.__file__)[0] + '/data/' 
model_path = '/home/arokem/data/ModelFits/MultiCanonicalTensor4/'

data_files = {1000:[make_data_set(
    '0009_01_DWI_2mm150dir_2x_b1000_aligned_trilin'),
    make_data_set(
    '0011_01_DWI_2mm150dir_2x_b1000_aligned_trilin')],
    2000:[make_data_set(
    '0005_01_DTI_2mm_150dir_2x_b2000_aligned_trilin'),
    make_data_set(
    '0007_01_DTI_2mm_150dir_2x_b2000_aligned_trilin')],
    4000:[make_data_set(
    '0005_01_DWI_2mm150dir_2x_b4000_aligned_trilin'),
    make_data_set(
    '0007_01_DWI_2mm150dir_2x_b4000_aligned_trilin')]}

# These are based on the calculations in GetADandRD
diffusivities = {1000: [dict(AD=1.7139, RD=0.3887),
                        dict(AD=1.6986, RD=0.3760)],
                2000: [dict(AD=1.4291, RD=0.3507),
                       dict(AD=1.4202, RD=0.3357)],
                4000: [dict(AD=0.8403, RD=0.2369),
                       dict(AD=0.8375, RD=0.2379)]}

for bval_idx, bval in enumerate([1000, 2000, 4000]):
    for f_idx, (dwi, bvecs, bvals) in enumerate(data_files[bval]):
        AD = diffusivities[bval][f_idx]['AD']
        RD = diffusivities[bval][f_idx]['RD']
        file_root = model_path + dwi.split('/')[-1].split('.')[0]
        print ("Working on %s"%dwi)
        brain_mask = ni.load(data_path + 'brainMask.nii.gz').get_data()
        brain_idx = np.array(np.where(brain_mask))

        vox_per_chunk = 1000
        n_chunks = int(np.round(np.sum(brain_mask)))/vox_per_chunk
        Model_list = []
        for chunk in range(n_chunks):
            if vox_per_chunk*chunk+1 > brain_idx.shape[-1]:
                this_idx = brain_idx[:,vox_per_chunk * chunk:]
            else:
                this_idx = brain_idx[:,
                            vox_per_chunk * chunk:vox_per_chunk*(chunk+1)]
            mask_array = np.zeros((81, 106, 76))
            mask_array[(this_idx[0], this_idx[1], this_idx[2])] = 1
        
            Model_list.append(ozm.MultiCanonicalTensorModel(dwi,
                                          bvecs,
                                          bvals,
                                          mask=mask_array,
                params_file='%s_%03d.nii.gz'%(file_root, chunk+1),
                                          radial_diffusivity=RD,
                                          axial_diffusivity=AD,
                                          verbose=False))


        # Farm *different* jobs to different engines
        for x1 in np.arange(0,n_chunks,n_engines):
            x2 = np.min([x1 + n_engines, len(Model_list)])
            Models = Model_list[x1:x2]
            chunks = np.arange(x1,x2)
            print "Sending chunks %s"%(chunks + 1)
            print(caller_func(Models, chunks, dview))
        
    
# Now recombine all the chunks into one file per scan: 
for bval_idx, bval in enumerate([1000, 2000, 4000]):
    for dwi, bvecs, bvals in data_files[bval]:
        file_root = model_path + dwi.split('/')[-1].split('.')[0]
        vol = ozu.nans((81,106,76,4))
        print ("Combining files for %s"%file_root)
        for chunk in range(n_chunks):
            if vox_per_chunk*chunk+1 > brain_idx.shape[-1]:
                this_idx = brain_idx[:,vox_per_chunk * chunk:]
            else:
                this_idx = brain_idx[:,
                            vox_per_chunk * chunk:vox_per_chunk*(chunk+1)]

            mask_array = np.zeros((81, 106, 76))
            mask_array[(this_idx[0], this_idx[1], this_idx[2])] = 1
            params_file = '%s_%03d.nii.gz'%(file_root, chunk+1)
            this_params = ni.load(params_file).get_data()
            mask_idx = np.where(mask_array)
            vol[mask_idx] = this_params[mask_idx]

        ni.Nifti1Image(vol, ni.load(params_file).get_affine()).to_filename(
        file_root + 'MultiCanonicalTensorModel.nii.gz')

