"""
This script calculates reliability of the PDD for different models and different b values

"""

import numpy as np
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

    file_root = []
    for dw in [dwi1, dwi2]:
        d,f = os.path.split(dw)
        file_root.append(d + '/' + f.split('.')[0])
    

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


    ## MultiCanonicalTensorModel1 = ozm.MultiCanonicalTensorModel(dwi1,
    ##                                         bvecs1,
    ##                                         bvals1,
    ##                                         #mode='normalize',
    ##                                         mask=brain_mask,
    ##                                         radial_diffusivity=RD1,
    ##                                         axial_diffusivity=AD1) 

    ## MultiCanonicalTensorModel2 = ozm.MultiCanonicalTensorModel(dwi2,
    ##                                         bvecs2,
    ##                                         bvals2,
    ##                                         #mode='normalize',
    ##                                         mask=brain_mask,
    ##                                         radial_diffusivity=RD2,
    ##                                         axial_diffusivity=AD2) 


    ## PointyMultiCanonicalTensorModel1 = ozm.MultiCanonicalTensorModel(dwi1,
    ##                                         bvecs1,
    ##                                         bvals1,
    ##                                         #mode='normalize',
    ##                                         params_file=(file_root[0] +
    ##                                 'PointyMultiCanonicalTensorModel.nii.gz'),
    ##                                         mask=brain_mask,
    ##                                         radial_diffusivity=0,
    ##                                         axial_diffusivity=AD1) 

    ## PointyMultiCanonicalTensorModel2 = ozm.MultiCanonicalTensorModel(dwi2,
    ##                                         bvecs2,
    ##                                         bvals2,
    ##                                         #mode='normalize',
    ##                                         params_file=(file_root[1] +
    ##                                 'PointyMultiCanonicalTensorModel.nii.gz'),
    ##                                         mask=brain_mask,
    ##                                         radial_diffusivity=0,
    ##                                         axial_diffusivity=AD2) 

    TensorModel1 = ozm.TensorModel(dwi1, 
                                   bvecs1, 
                                   bvals1, 
                                   mask=brain_mask)

    TensorModel2 = ozm.TensorModel(dwi2, 
                                   bvecs2, 
                                   bvals2, 
                                   mask=brain_mask)


    ## SparseDeconvolutionModel1 = ozm.SparseDeconvolutionModel(dwi1,
    ##                                         bvecs1,
    ##                                         bvals1,
    ##                                         axial_diffusivity=AD1,
    ##                                         radial_diffusivity=RD1,
    ##                                         mask=brain_mask)

    ## SparseDeconvolutionModel2 = ozm.SparseDeconvolutionModel(dwi2,
    ##                                         bvecs2,
    ##                                         bvals2,
    ##                                         axial_diffusivity=AD2,
    ##                                         radial_diffusivity=RD2,
    ##                                         mask=brain_mask)

    ## SphereModel1 = ozm.SphereModel(dwi1,
    ##                              bvecs1,
    ##                              bvals1,
    ##                              mask=brain_mask)

    ## SphereModel2 = ozm.SphereModel(dwi2,
    ##                                bvecs2,
    ##                                bvals2,
    ##                                mask=brain_mask)


    ModelFest = zip([
        TensorModel1,
        CanonicalTensorModel1,
        #MultiCanonicalTensorModel1,
        #SparseDeconvolutionModel1,
        #SphereModel1,
        #PointyMultiCanonicalTensorModel1
        ],
                     [
        TensorModel2,
        CanonicalTensorModel2,
        #MultiCanonicalTensorModel2,
        #SparseDeconvolutionModel2,
        #SphereModel2,
        #PointyMultiCanonicalTensorModel2
        ],
                     [
        'TensorModel',
        'CanonicalTensorModel',
        #'MultiCanonicalTensorModel',
        #'SparseDeconvolutionModel',
        #'SphereModel',
        #'PointyMultiCanonicalTensorModel',
                         ])

    for Model1,Model2,model_name in ModelFest:      
        PDD1 = Model1.principal_diffusion_direction
        PDD2 = Model2.principal_diffusion_direction
        PDD1_flat = PDD1[Model1.mask]
        PDD2_flat = PDD2[Model2.mask]
        n_vox = PDD1_flat.shape[0]
        angles_flat = np.empty(n_vox)
        prog_bar = viz.ProgressBar(n_vox)
        for vox in xrange(n_vox):
            
            angles_flat[vox] = ozu.vector_angle(PDD1_flat[vox],
                                                PDD2_flat[vox])
            prog_bar.animate(vox)
        angles = ozu.nans(PDD1.shape[:3])
        angles[PDD1.mask] = angles_flat
    


