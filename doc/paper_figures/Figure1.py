# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import os
#os.environ['DISPLAY'] = 'localhost:99'

# <codecell>

import os
import tempfile
from IPython.display import Image, display

import nibabel as ni

import osmosis as oz
import osmosis.model.dti as dti
import osmosis.model.sparse_deconvolution as ssd


import osmosis.viz.maya as maya
reload(maya)
import osmosis.viz.mpl as mpl

import osmosis.utils as ozu
import osmosis.volume as ozv
import osmosis.io as oio
oio.data_path = '/biac4/wandell/biac2/wandell6/data/arokem/osmosis'
import osmosis.tensor as ozt

# <codecell>

subject = 'FP'
data_1k_1, data_1k_2 = oio.get_dwi_data(1000, subject)
data_2k_1, data_2k_2 = oio.get_dwi_data(2000, subject)
data_4k_1, data_4k_2 = oio.get_dwi_data(4000, subject)

# <codecell>

# A corpus callosum voxel:
vox_idx = (40, 74, 34)

mask = np.zeros(ni.load(data_1k_1[0]).shape[:3])
mask[vox_idx[0]-1:vox_idx[0]+1,
     vox_idx[1]-1:vox_idx[1]+1,
     vox_idx[2]-1:vox_idx[2]+1]=1

# <codecell>

TM_1k_1 = dti.TensorModel(*data_1k_1, mask=mask, params_file='temp')
TM_1k_2 = dti.TensorModel(*data_1k_2, mask=mask, params_file='temp')
TM_2k_1 = dti.TensorModel(*data_2k_1, mask=mask, params_file='temp')
TM_2k_2 = dti.TensorModel(*data_2k_2, mask=mask, params_file='temp')
TM_4k_1 = dti.TensorModel(*data_4k_1, mask=mask, params_file='temp')
TM_4k_2 = dti.TensorModel(*data_4k_2, mask=mask, params_file='temp')

# <codecell>

%gui wx
# Interpolate the signal from the measurement b-vectors using radial basis functions and display in 3D:
fig = maya.plot_signal_interp(TM_1k_1.bvecs[:,TM_1k_1.b_idx], TM_1k_1.relative_signal[vox_idx], cmap='hot', origin=[0,0,0], vmin=0, vmax=1)
fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_1.b_idx], TM_2k_1.relative_signal[vox_idx], cmap='hot', origin=[0,0,2], vmin=0, vmax=1, figure=fig, offset=-2)
# Put it into a temporary file:

fn = '%s.png'%tempfile.NamedTemporaryFile().name 
#orientation_axes = mlab.orientation_axes(xlabel='', ylabel='').axes
fig = maya.plot_signal_interp(TM_4k_1.bvecs[:,TM_4k_1.b_idx], TM_4k_1.relative_signal[vox_idx], cmap='hot', origin=[0,0,4], colorbar=True, vmin=0, vmax=1, figure=fig, offset=-4, file_name=fn, roll=90)

# <codecell>

i = Image(filename=fn)
display(i)

# <codecell>

# Make figures for a movie: 

#for ang in range(360):
#    fn = '/home/arokem/Dropbox/osmosis_paper_figures/movie_figures_sig_2k/frame%03d.png'%ang
#    if not os.path.exists(fn):
#        fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_1.b_idx], TM_2k_1.relative_signal[vox_idx], cmap='hot', colorbar=True, vmin=0, vmax=1, elevation=ang, file_name=fn)

# <codecell>

#for ang in range(360):
#    fn = '/home/arokem/Dropbox/osmosis_paper_figures/movie_figures_sig_2k_w_pts/frame%03d.png'%ang
#    if not os.path.exists(fn):
#        fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_1.b_idx], TM_2k_1.relative_signal[vox_idx], cmap='hot', colorbar=True, vmin=0, vmax=1, elevation=ang, file_name=fn, points=True,scale_points=False)

# <codecell>

# Interpolate the signal from the measurement b-vectors using radial basis functions and display in 3D:
fig = maya.plot_signal_interp(TM_1k_1.bvecs[:,TM_1k_1.b_idx], TM_1k_1.fit[vox_idx]/TM_1k_1.S0[vox_idx], origin=[0,0,0], cmap='hot', vmin=0, vmax=1)
fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_1.b_idx], TM_2k_1.fit[vox_idx]/TM_1k_1.S0[vox_idx], origin=[0,0,2], cmap='hot', figure=fig, offset=-2, vmin=0, vmax=1)
# Put it into a temporary file:
fn = '%s.png'%tempfile.NamedTemporaryFile().name 
#orientation_axes = mlab.orientation_axes(xlabel='', ylabel='').axes
fig = maya.plot_signal_interp(TM_4k_1.bvecs[:,TM_4k_1.b_idx], TM_4k_1.fit[vox_idx]/TM_4k_1.S0[vox_idx], origin=[0,0,4], cmap='hot', colorbar=True, figure=fig, offset=-4, file_name=fn, roll=90, vmin=0, vmax=1)

# <codecell>

i = Image(filename=fn)
display(i)

# <codecell>

ten_1k = ozt.tensor_from_eigs(TM_1k_1.model_params[vox_idx][3:].reshape(3,3), 
                           TM_1k_1.model_params[vox_idx][:3], 
                           TM_1k_1.bvecs[:, TM_1k_1.b_idx], TM_1k_1.bvals[TM_1k_1.b_idx])

fig = maya.plot_tensor_3d(ten_1k, mode='ADC', cmap='hot', origin=[0,0,0], colorbar=True, vmin=0, vmax=2.5)

ten_2k = ozt.tensor_from_eigs(TM_2k_1.model_params[vox_idx][3:].reshape(3,3), 
                           TM_2k_1.model_params[vox_idx][:3], 
                           TM_2k_1.bvecs[:, TM_2k_1.b_idx], TM_2k_1.bvals[TM_2k_1.b_idx])

fig = maya.plot_tensor_3d(ten_2k, mode='ADC', cmap='hot', origin=[0,0,2], vmin=0, vmax=2.5, figure=fig, offset=-2.5)


ten_4k = ozt.tensor_from_eigs(TM_4k_1.model_params[vox_idx][3:].reshape(3,3), 
                           TM_4k_1.model_params[vox_idx][:3], 
                           TM_4k_1.bvecs[:, TM_4k_1.b_idx], TM_4k_1.bvals[TM_4k_1.b_idx])

fn = '%s.png'%tempfile.NamedTemporaryFile().name 
fig = maya.plot_tensor_3d(ten_4k, mode='ADC', cmap='hot', origin=[0,0,4], vmin=0, vmax=2.5, figure=fig, offset=-5, file_name=fn, roll=90)


# <codecell>

i = Image(filename=fn, width=1280, height=1024)
display(i)

# <codecell>

ten_1k = ozt.tensor_from_eigs(TM_1k_1.model_params[vox_idx][3:].reshape(3,3), 
                           TM_1k_1.model_params[vox_idx][:3], 
                           TM_1k_1.bvecs[:, TM_1k_1.b_idx], TM_1k_1.bvals[TM_1k_1.b_idx])

fig = maya.plot_tensor_3d(ten_1k, mode='ellipse', cmap='Greens', colorbar=True)
ten_2k = ozt.tensor_from_eigs(TM_2k_1.model_params[vox_idx][3:].reshape(3,3), 
                           TM_2k_1.model_params[vox_idx][:3], 
                           TM_2k_1.bvecs[:, TM_2k_1.b_idx], TM_2k_1.bvals[TM_2k_1.b_idx])

fig = maya.plot_tensor_3d(ten_2k, mode='ellipse', cmap='Greens',  offset=-2.5, figure=fig)


ten_4k = ozt.tensor_from_eigs(TM_4k_1.model_params[vox_idx][3:].reshape(3,3), 
                           TM_4k_1.model_params[vox_idx][:3], 
                           TM_4k_1.bvecs[:, TM_4k_1.b_idx], TM_4k_1.bvals[TM_4k_1.b_idx])

fn = '%s.png'%tempfile.NamedTemporaryFile().name 
fig = maya.plot_tensor_3d(ten_4k, mode='ellipse', cmap='Greens', offset=-5, figure=fig, file_name=fn, roll=90)

# <codecell>

i = Image(filename=fn, width=1280, height=1024)
display(i)

# <codecell>

fig, axes = plt.subplots(1,3,squeeze=True)
axes[0].set_ylabel(r'$\frac{S}{S_0}$')
for ax, models in zip(axes,([TM_1k_1,TM_1k_2],[TM_2k_1, TM_2k_2],[TM_4k_1, TM_4k_2])):
    ax.scatter(models[0].relative_signal[vox_idx], models[1].relative_signal[vox_idx])
    ax.grid()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\frac{S}{S_0}$')
    
fig.set_size_inches([10,10])

# <codecell>

fig, axes = plt.subplots(1,3,squeeze=True)
axes[0].set_ylabel(r'Predicted $\frac{S}{S_0}$')
for ax, models in zip(axes,([TM_1k_1,TM_1k_2],[TM_2k_1, TM_2k_2],[TM_4k_1, TM_4k_2])):
    ax.scatter(models[0].relative_signal[vox_idx], models[1].fit[vox_idx]/models[1].S0[vox_idx])
    ax.grid()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')
    ax.set_xlabel(r'Measured $\frac{S}{S_0}$')
    
fig.set_size_inches([10,10])

# <codecell>


