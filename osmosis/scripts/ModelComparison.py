"""
This script calculates relative RMSE for different models and different b values

"""


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

from data import *

for bval_idx, bval in enumerate([1000, 2000, 4000]):
    AD1,RD1 = diffusivities[bval][0]['AD'], diffusivities[bval][0]['RD']
    AD2,RD2 = diffusivities[bval][1]['AD'], diffusivities[bval][1]['RD']
    dwi1, bvecs1, bvals1 = data_files[bval][0]
    dwi2, bvecs2, bvals2 = data_files[bval][1]

    
    CanonicalTensorModel1 = ozm.CanonicalTensorModel(dwi1,
                                        bvecs1,
                                        bvals1,
                                        #mode='normalize',
                                        mask = brain_mask,
                                        radial_diffusivity=RD1,
                                        axial_diffusivity=AD1)
    
    CanonicalTensorModel2 = ozm.CanonicalTensorModel(dwi2,
                                        bvecs2,
                                        bvals2,
                                        #mode='normalize',
                                        mask = brain_mask,
                                        radial_diffusivity=RD2,
                                        axial_diffusivity=AD2) 


    CanonicalTensorModel1 = ozm.CanonicalTensorModel(dwi1,
                                        bvecs1,
                                        bvals1,
                                        #mode='normalize',
                                        mask = brain_mask,
                                        radial_diffusivity=RD1,
                                        axial_diffusivity=AD1)
    
    CanonicalTensorModel2 = ozm.CanonicalTensorModel(dwi2,
                                        bvecs2,
                                        bvals2,
                                        #mode='normalize',
                                        mask = brain_mask,
                                        radial_diffusivity=RD2,
                                        axial_diffusivity=AD2) 

    MultiCanonicalTensorModel1 = ozm.MultiCanonicalTensorModel(dwi1,
                                            bvecs1,
                                            bvals1,
                                            #mode='normalize',
                                            mask=brain_mask,
                                            radial_diffusivity=RD1,
                                            axial_diffusivity=AD1) 

    MultiCanonicalTensorModel2 = ozm.MultiCanonicalTensorModel(dwi2,
                                            bvecs2,
                                            bvals2,
                                            #mode='normalize',
                                            mask=brain_mask,
                                            radial_diffusivity=RD2,
                                            axial_diffusivity=AD2) 

    TensorModel1 = ozm.TensorModel(dwi1, 
                                   bvecs1, 
                                   bvals1, 
                                   mask=brain_mask)

    TensorModel2 = ozm.TensorModel(dwi2, 
                                   bvecs2, 
                                   bvals2, 
                                   mask=brain_mask)

    SparseDeconvolutionModel1 = ozm.SparseDeconvolutionModel(dwi1,
                                            bvecs1,
                                            bvals1,
                                            axial_diffusivity=AD1,
                                            radial_diffusivity=RD1,
                                            mask=brain_mask)

    SparseDeconvolutionModel2 = ozm.SparseDeconvolutionModel(dwi2,
                                            bvecs2,
                                            bvals2,
                                            axial_diffusivity=AD2,
                                            radial_diffusivity=RD2,
                                            mask=brain_mask)

    SphereModel1 = ozm.SphereModel(dwi1,
                                 bvecs1,
                                 bvals1,
                                 mask=brain_mask)

    SphereModel2 = ozm.SphereModel(dwi2,
                                   bvecs2,
                                   bvals2,
                                   mask=brain_mask)


    ModelFest = zip([TensorModel1,CanonicalTensorModel1,
                     MultiCanonicalTensorModel1,SparseDeconvolutionModel1,
                     SphereModel1],
                     [TensorModel2,CanonicalTensorModel2,
                      MultiCanonicalTensorModel2,SparseDeconvolutionModel2,
                      SphereModel2],
                     ['TensorModel', 'CanonicalTensorModel',
                      'MultiCanonicalTensorModel','SparseDeconvolutionModel',
                         'SphereModel'])


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
        

