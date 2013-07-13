# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import os
#os.environ['DISPLAY'] = 'localhost:99'

#from IPython.display import Image, display

# <codecell>

import os
import tempfile
from IPython.display import Image, display

import nibabel as ni

import osmosis as oz
import osmosis.model.analysis as ozm
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

vox_idx = (53, 43, 47)
mask = np.zeros(ni.load(data_1k_1[0]).shape[:3])
mask[vox_idx[0]:vox_idx[0]+2, vox_idx[1], vox_idx[2]]=1

# <codecell>

TM_1k_1 = dti.TensorModel(*data_1k_1, mask=mask, params_file='temp')
TM_1k_2 = dti.TensorModel(*data_1k_2, mask=mask, params_file='temp')
TM_2k_1 = dti.TensorModel(*data_2k_1, mask=mask, params_file='temp')
TM_2k_2 = dti.TensorModel(*data_2k_2, mask=mask, params_file='temp')
TM_4k_1 = dti.TensorModel(*data_4k_1, mask=mask, params_file='temp')
TM_4k_2 = dti.TensorModel(*data_4k_2, mask=mask, params_file='temp')

alpha = 0.0005
l1_ratio = 0.6

solver_params = dict(alpha=alpha,
                     l1_ratio=l1_ratio,
                     fit_intercept=False,
                     positive=True)


ad_rd = oio.get_ad_rd(subject, 1000)
SD_1k_1 = ssd.SparseDeconvolutionModel(*data_1k_1, mask=mask, solver_params=solver_params, params_file='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
SD_1k_2 = ssd.SparseDeconvolutionModel(*data_1k_2, mask=mask, solver_params=solver_params, params_file='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])
ad_rd = oio.get_ad_rd(subject, 2000)
SD_2k_1 = ssd.SparseDeconvolutionModel(*data_2k_1, mask=mask, solver_params=solver_params, params_file='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
SD_2k_2 = ssd.SparseDeconvolutionModel(*data_2k_2, mask=mask, solver_params=solver_params, params_file='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])
ad_rd = oio.get_ad_rd(subject, 4000)
SD_4k_1 = ssd.SparseDeconvolutionModel(*data_4k_1, mask=mask, solver_params=solver_params, params_file='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
SD_4k_2 = ssd.SparseDeconvolutionModel(*data_4k_2, mask=mask, solver_params=solver_params, params_file='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])

# <codecell>

ten_rrmse_1k = ozm.cross_predict(TM_1k_1, TM_1k_2)
sfm_rrmse_1k = ozm.cross_predict(SD_1k_1, SD_1k_2)

ten_rrmse_2k = ozm.cross_predict(TM_2k_1, TM_2k_2)
sfm_rrmse_2k = ozm.cross_predict(SD_2k_1, SD_2k_2)

ten_rrmse_4k = ozm.cross_predict(TM_4k_1, TM_4k_2)
sfm_rrmse_4k = ozm.cross_predict(SD_4k_1, SD_4k_2)

# <codecell>

%gui wx

fig = maya.plot_signal_interp(TM_4k_1.bvecs[:,TM_4k_1.b_idx], TM_4k_1.relative_signal[vox_idx], cmap='hot', colorbar=True, vmin=0, vmax=1)#, figure=fig, offset=-4, roll=90, file_name=fn)
fig = maya.plot_signal_interp(TM_4k_1.bvecs[:,TM_4k_2.b_idx], TM_4k_2.relative_signal[vox_idx], cmap='hot', colorbar=True, figure=fig, vmin=0, vmax=1, origin = [0,0,-1])#, figure=fig, offset=-4, roll=90, file_name=fn)
fig = maya.plot_signal_interp(TM_4k_1.bvecs[:,TM_4k_1.b_idx], TM_4k_1.fit[vox_idx]/TM_4k_1.S0[vox_idx], cmap='hot', colorbar=True, figure=fig, origin = [0,0,1], roll=90, vmin=0, vmax=1)#, file_name=fn)
fig = maya.plot_signal_interp(SD_4k_1.bvecs[:,SD_4k_1.b_idx], SD_4k_1.fit[vox_idx]/SD_4k_1.S0[vox_idx], cmap='hot', colorbar=True, figure=fig, origin = [0,0,-2], roll=90, vmin=0, vmax=1)#, file_name=fn)

print(TM_4k_1.fractional_anisotropy[vox_idx])
print(ten_rrmse_4k[vox_idx])
print(sfm_rrmse_4k[vox_idx])

# <codecell>

%gui wx
fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_1.b_idx], TM_2k_1.relative_signal[vox_idx], cmap='hot', colorbar=True, vmin=0, vmax=1)#, figure=fig, offset=-4, roll=90, file_name=fn)
fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_2.b_idx], TM_2k_2.relative_signal[vox_idx], cmap='hot', colorbar=True, figure=fig, vmin=0, vmax=1, origin = [0,0,-1.1])#, figure=fig, offset=-4, roll=90, file_name=fn)
fig = maya.plot_signal_interp(TM_2k_1.bvecs[:,TM_2k_1.b_idx], TM_2k_1.fit[vox_idx]/TM_2k_1.S0[vox_idx], cmap='hot', colorbar=True, figure=fig, origin = [0,0,1.1], roll=90, vmin=0, vmax=1)#, file_name=fn)
fig = maya.plot_signal_interp(SD_2k_1.bvecs[:,SD_2k_1.b_idx], SD_2k_1.fit[vox_idx]/SD_2k_1.S0[vox_idx], cmap='hot', colorbar=True, figure=fig, origin = [0,0,-2.1], roll=90, vmin=0, vmax=1)#, file_name=fn)

#print(TM_2k_1.fractional_anisotropy[vox_idx])
#print(ten_rrmse_2k[vox_idx])
#print(sfm_rrmse_2k[vox_idx])

# <codecell>

%gui wx
fig = maya.plot_signal_interp(TM_1k_1.bvecs[:,TM_1k_1.b_idx], TM_1k_1.relative_signal[vox_idx], cmap='hot', colorbar=True, vmin=0, vmax=1)#, figure=fig, offset=-4, roll=90, file_name=fn)
fig = maya.plot_signal_interp(TM_1k_1.bvecs[:,TM_1k_1.b_idx], TM_1k_2.relative_signal[vox_idx], cmap='hot', colorbar=True, figure=fig, vmin=0, vmax=1, origin = [0,0,-1.1])#, figure=fig, offset=-4, roll=90, file_name=fn)
fig = maya.plot_signal_interp(TM_1k_1.bvecs[:,TM_1k_1.b_idx], TM_1k_1.fit[vox_idx]/TM_1k_1.S0[vox_idx], cmap='hot', colorbar=True, figure=fig, origin = [0,0,1.1], roll=90, vmin=0, vmax=1)#, file_name=fn)
fig = maya.plot_signal_interp(SD_1k_1.bvecs[:,SD_1k_1.b_idx], SD_1k_1.fit[vox_idx]/SD_1k_1.S0[vox_idx], cmap='hot', colorbar=True, figure=fig, origin = [0,0,-2.1], roll=90, vmin=0, vmax=1)#, file_name=fn)

#print(TM_1k_1.fractional_anisotropy[vox_idx])
#print(ten_rrmse_1k[vox_idx])
#print(sfm_rrmse_1k[vox_idx])

# <codecell>


