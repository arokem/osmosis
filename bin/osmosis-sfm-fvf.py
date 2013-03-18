#!/usr/bin/env python
import os
import tempfile
import argparse as arg

import numpy as np
import nibabel as nib
import osmosis.model.sparse_deconvolution as sfm

parser = arg.ArgumentParser('Calculate the sum of weights in the sparse fascicle model')

parser.add_argument('dwi_file', action='store', metavar='File', 
                    help='DWI file (.nii.gz)')

parser.add_argument('bvecs_file',action='store', metavar='File', 
                   help='Bvecs file (FSL format)')

parser.add_argument('bvals_file', action='store', metavar='File',
                    help='Bvals file (FSL format)')

parser.add_argument('out_file', action='store', metavar='File',
                    help='Output file name (.nii.gz)')

parser.add_argument('--mask_file', action='store', metavar='File',
                    help='Mask file (only the voxels within the binary mask will be analyzed (.nii.gz; default: analyze all) ',
                    default=None)

parser.add_argument('--params_output', action='store', metavar='File',
                    help='If you want to save the model parameters as a nifti file, provide a file-name here (default: do not save params)', default=None)


parser.add_argument('--alpha', action='store', metavar='Float',
                    help='Regularization parameter : how strong should regularization be (default: 0.0005)', default=0.0005)

parser.add_argument('--l1_ratio', action='store', metavar='Float',
                    help='Regularization parameter : how L1 weighted should regularization be (default: 0.6)', default=0.6)

params = parser.parse_args()


if __name__ == "__main__":

    solver_params = dict(l1_ratio=params.l1_ratio,
                         alpha=params.alpha,
                         fit_intercept=False,
                         positive=True)
    if args.params_output is None:
        params_file = 'temp'
    else:
        params_file = params.params_output
        
    Model = sfm.SparseDeconvolutionModel(params.dwi_file,
                                         params.bvecs_file,
                                         params.bvals_file,
                                         mask=params.mask_file,
                                         solver_params=solver_params,
                                         params_file=params_file)

    # Do it and save: 
    fvf = nib.Nifti1Image(np.sum(Model.model_params, -1),
                         Model.affine).to_filename(params.out_file)

