#!/usr/bin/env python
import os
import tempfile
import argparse as arg

import numpy as np
import nibabel as nib
from mayavi import mlab

import osmosis.viz.maya as maya

parser = arg.ArgumentParser('Visualize the data in a single voxel')

parser.add_argument('dwi_file', action='store', metavar='File', 
                    help='DWI file (.nii.gz)')

parser.add_argument('bvecs_file',action='store', metavar='File', 
                   help='Bvecs file (FSL format)')

parser.add_argument('bvals_file', action='store', metavar='File',
                    help='Bvals file (FSL format)')

parser.add_argument('x', action='store', metavar='int',
                   help='the voxel x coordinate')

parser.add_argument('y', action='store', metavar='int', 
                   help='the voxel y coordinate')

parser.add_argument('z', action='store', metavar='int',
                   help='the voxel z coordinate')

parser.add_argument('--mode', action='store', dest='mode', metavar='str', 
                    default='signal',choices=['dti', 'sfm', 'signal'],
                    help="What to visualize ")

parser.add_argument('--out', action='store', dest='out', metavar='File', 
                    default=None, choices=['dti', 'sfm', 'signal'],
                    help="Save to file")

params = parser.parse_args()


if __name__ == "__main__":
    vox_idx = (params.x,params.y,params.z)    
    data_shape = nib.load(params.dwi_file).shape
    mask = np.zeros(data_shape[:3], dtype=bool)
    mask[vox_idx] = True
    kwargs = dict(cmap='hot')

    if params.mode == 'sfm':
        import osmosis.model.sparse_deconvolution as sfm
        l1_ratio = 0.6
        alpha = 0.0005 
        solver_params = dict(l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False,
                             positive=True)

        Model = sfm.SparseDeconvolutionModel(params.dwi_file,
                                params.bvecs_file,
                                params.bvals_file,
                                mask=mask,
                                solver_params=solver_params,
                                params_file='temp')

        signal = Model.model_params[vox_idx]
        plotter = maya.plot_odf_interp
        
    elif params.mode == 'dti':
        import osmosis.model.dti as dti
        Model = dti.TensorModel(params.dwi_file,
                                params.bvecs_file,
                                params.bvals_file,
                                mask=mask,
                                params_file='temp')
        
        signal = Model.model_adc[vox_idx]
        plotter = maya.plot_signal_interp

    elif params.mode == 'signal': 
        import osmosis.model.base as ozm
        Model = ozm.DWI(params.dwi_file, params.bvecs_file, params.bvals_file,
                        mask=mask)
        signal = Model.signal[vox_idx]/float(Model.S0[vox_idx])
        plotter = maya.plot_signal_interp
        kwargs.update(vmin=0, vmax=1)

    bvecs = Model.bvecs[:,Model.b_idx]

    fig = plotter(bvecs,signal, **kwargs)

    mlab.show()
