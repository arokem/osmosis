"""
Generate relative RMSE images for the SH model
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import nibabel as ni

import osmosis.io as io
import osmosis as oz
import osmosis.model as ozm
reload(ozm)
import osmosis.tensor as ozt
reload(ozt)
import osmosis.viz as viz
import osmosis.utils as ozu

from data import *

for bval_idx, bval in enumerate([1000, 2000, 4000]):
    AD1,RD1 = diffusivities[bval][0]['AD'], diffusivities[bval][0]['RD']
    AD2,RD2 = diffusivities[bval][1]['AD'], diffusivities[bval][1]['RD']
    dwi1, bvecs1, bvals1 = data_files[bval][0]
    dwi2, bvecs2, bvals2 = data_files[bval][1]

    csd_coeffs = []
    response_files = []
    for dw in [dwi1, dwi2]:
        d,f = os.path.split(dw)
        csd_coeffs.append(d + '/' + f.split('.')[0] + '_CSD.nii.gz')
        response_files.append(d + '/' + f.split('.')[0] + '_ER.mif')

    SHM1 = ozm.SphericalHarmonicsModel(dwi1, bvecs1, bvals1,
                                  mask = brain_mask , 
                                  model_coeffs = csd_coeffs[0],
                                  response_file = response_files[0],
                                  #axial_diffusivity=AD1,
                                  #radial_diffusivity=RD1
                                  )

    SHM2 = ozm.SphericalHarmonicsModel(dwi2, bvecs2, bvals2,
                                  #mask = mask_array,
                                  mask = brain_mask , 
                                  model_coeffs = csd_coeffs[1],
                                  response_file = response_files[1],
                                  #axial_diffusivity=AD2,
                                  #radial_diffusivity=RD2
                                  )

    rmse_file_name = '%s%s_relative_rmse_b%s.nii.gz'%(data_path,
                                                     'SphericalHarmonicsModel',
                                                      bval)
    if not os.path.isfile(rmse_file_name):
        relative_rmse = ozm.relative_rmse(SHM1, SHM2)
        io.nii_from_volume(relative_rmse,
                       rmse_file_name,
                       ni.load(dwi1).get_affine())


