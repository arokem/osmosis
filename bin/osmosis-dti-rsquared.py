#!/usr/bin/env python
import os
import tempfile
import argparse as arg

import numpy as np
import nibabel as nib
import osmosis.model.dti as dti
import osmosis.model.analysis as ana

parser = arg.ArgumentParser(description='Calculate the cross-validation coefficient of determination for the DTI model')

parser.add_argument('dwi_file1', action='store', metavar='File', 
                    help='DWI file (.nii.gz)')

parser.add_argument('bvecs_file1',action='store', metavar='File', 
                   help='Bvecs file (FSL format)')

parser.add_argument('bvals_file1', action='store', metavar='File',
                    help='Bvals file (FSL format)')

parser.add_argument('dwi_file2', action='store', metavar='File', 
                    help='DWI file (.nii.gz)')

parser.add_argument('bvecs_file2',action='store', metavar='File', 
                   help='Bvecs file (FSL format)')

parser.add_argument('bvals_file2', action='store', metavar='File',
                    help='Bvals file (FSL format)')

parser.add_argument('out_file', action='store', metavar='File',
                    help='Output file name (.nii.gz)')

parser.add_argument('--mask_file', action='store', metavar='File',
                    help='Mask file (only the voxels within the binary mask will be analyzed (.nii.gz; default: analyze all) ',
                    default=None)

params = parser.parse_args()


if __name__ == "__main__":
     
    Model1 = dti.TensorModel(params.dwi_file1,
                                         params.bvecs_file1,
                                         params.bvals_file1,
                                         mask=params.mask_file,
                                         params_file='temp')

    Model2 = dti.TensorModel(params.dwi_file2,
                                         params.bvecs_file2,
                                         params.bvals_file2,
                                         mask=params.mask_file,
                                         params_file='temp')
    
    # Do it and save: 
    nib.Nifti1Image(ana.rsquared(Model1, Model2),
                    Model1.affine).to_filename(params.out_file)

