import nibabel as ni
import os
import osmosis as oz
import osmosis.model as ozm
import osmosis.viz as viz
import osmosis.utils as ozu
import osmosis.volume as ozv
import osmosis.io as io
reload(ozm)
reload(ozu)

data_path = os.path.split(oz.__file__)[0] + '/data/'

def make_data_set(root):
    """
    Create the full paths to the data given a root
    """ 
    exts = ['.nii.gz', '.bvecs', '.bvals']
    dwi, bvecs, bvals = [data_path + root + ext for ext in exts]
    return dwi, bvecs, bvals 

data_files = {1000:[make_data_set(
    '0009_01_DWI_2mm150dir_2x_b1000_aligned_trilin'),
    make_data_set(
    '0011_01_DWI_2mm150dir_2x_b1000_aligned_trilin')],
    2000:[make_data_set(
    '0005_01_DTI_2mm_150dir_2x_b2000_aligned_trilin'),
    make_data_set(
    '0007_01_DTI_2mm_150dir_2x_b2000_aligned_trilin')],
    4000:[make_data_set(
    '0005_01_DTI_2mm_150dir_2x_b4000_aligned_trilin'),
    make_data_set(
    '0007_01_DTI_2mm_150dir_2x_b4000_aligned_trilin')]}

# These are based on the calculations in GetADandRD
diffusivities = {1000: [dict(AD=1.7139, RD=0.3887),
                        dict(AD=1.6986, RD=0.3760)],
                2000: [dict(AD=1.4291, RD=0.3507),
                       dict(AD=1.4202, RD=0.3357)],
                4000: [dict(AD=0.8403, RD=0.2369),
                       dict(AD=0.8375, RD=0.2379)]}


for bval_idx, bval in enumerate([1000, 2000, 4000]):
    AD1,RD1 = diffusivities[bval][0]['AD'], diffusivities[bval][0]['RD']
    AD2,RD2 = diffusivities[bval][1]['AD'], diffusivities[bval][1]['RD']
    dwi1, bvecs1, bvals1 = data_files[bval][0]
    dwi2, bvecs2, bvals2 = data_files[bval][1]
    
    CanonicalTensorModel1 = ozm.CanonicalTensorModel(dwi1,
                                        bvecs1,
                                        bvals1,
                                        mask = data_path + 'brainMask.nii.gz',
                                        radial_diffusivity=RD1,
                                        axial_diffusivity=AD1)
    
    CanonicalTensorModel2 = ozm.CanonicalTensorModel(dwi2,
                                        bvecs2,
                                        bvals2,
                                        mask = data_path + 'brainMask.nii.gz',
                                        radial_diffusivity=RD2,
                                        axial_diffusivity=AD2) 

    MultiCanonicalTensorModel1 = ozm.MultiCanonicalTensorModel(dwi1,
                                            bvecs1,
                                            bvals1,
                                            mask=data_path + 'brainMask.nii.gz',
                                            radial_diffusivity=RD1,
                                            axial_diffusivity=AD1) 

    MultiCanonicalTensorModel2 = ozm.MultiCanonicalTensorModel(dwi2,
                                            bvecs2,
                                            bvals2,
                                            mask=data_path + 'brainMask.nii.gz',
                                            radial_diffusivity=RD2,
                                            axial_diffusivity=AD2) 

    TensorModel1 = ozm.TensorModel(dwi1, 
                                   bvecs1, 
                                   bvals1, 
                                   mask=data_path + 'brainMask.nii.gz')

    TensorModel2 = ozm.TensorModel(dwi2, 
                                   bvecs2, 
                                   bvals2, 
                                   mask=data_path + 'brainMask.nii.gz')

    SparseDeconvolutionModel1 = ozm.SparseDeconvolutionModel(dwi1,
                                            bvecs1,
                                            bvals1,
                                            axial_diffusivity=AD1,
                                            radial_diffusivity=RD1,
                                            mask=data_path + 'brainMask.nii.gz')

    SparseDeconvolutionModel2 = ozm.SparseDeconvolutionModel(dwi2,
                                            bvecs2,
                                            bvals2,
                                            axial_diffusivity=AD2,
                                            radial_diffusivity=RD2,
                                            mask=data_path + 'brainMask.nii.gz')

    ModelFest = zip([TensorModel1,CanonicalTensorModel1,
                     MultiCanonicalTensorModel1,SparseDeconvolutionModel1],
                     [TensorModel2,CanonicalTensorModel2,
                      MultiCanonicalTensorModel2,SparseDeconvolutionModel2],
                     ['TensorModel', 'CanonicalTensorModel',
                      'MultiCanonicalTensorModel','SparseDeconvolutionModel'])


    # Compute the relative RMSE for each model: 
    relative_rmse = []
    for Model1,Model2,model_name in ModelFest:      
        rmse_file_name = '%s%s_relative_rmse_b%s.nii.gz'%(data_path,
                                                          model_name,
                                                          bval)
        # Only do this if you have to:
        if not os.path.isfile(rmse_file_name):
            relative_rmse = ozm.relative_rmse(Model1, Model2)
            io.nii_from_volume(relative_rmse,
                               rmse_file_name,
                               ni.load(dwi1).get_affine())
        

