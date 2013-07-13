# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import os 
#os.environ['DISPLAY'] = 'localhost:14.0'

# <codecell>

import os
import tempfile

from IPython.display import Image, display
import nibabel as ni

import osmosis as oz
import osmosis.viz.maya as viz
import osmosis.utils as ozu
import osmosis.io as oio
import osmosis.model.sparse_deconvolution as ssd
import osmosis.model.analysis as oza

# <codecell>

subject = 'FP'
data_1k_1, data_1k_2 = oio.get_dwi_data(1000, subject)
data_2k_1, data_2k_2 = oio.get_dwi_data(2000, subject)
data_4k_1, data_4k_2 = oio.get_dwi_data(4000, subject)

# <codecell>

wm_mask = np.zeros(ni.load(data_1k_1[0]).shape[:3])
wm_nifti = ni.load(oio.data_path + '%s/%s_wm_mask.nii.gz'%(subject, subject)).get_data()
wm_mask[np.where(wm_nifti==1)] = 1

# <codecell>

# This is the best according to rRMSE across bvals: 
l1_ratio = 0.8
alpha = 0.0005 
solver_params = dict(l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False, positive=True)

# <codecell>

ad_rd = oio.get_ad_rd(subject, 1000)
SD_1k_1 = ssd.SparseDeconvolutionModel(*data_1k_1, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_1k_2 = ssd.SparseDeconvolutionModel(*data_1k_2, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
ad_rd = oio.get_ad_rd(subject, 2000)
SD_2k_1 = ssd.SparseDeconvolutionModel(*data_2k_1, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_2k_2 = ssd.SparseDeconvolutionModel(*data_2k_2, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
ad_rd = oio.get_ad_rd(subject, 4000)
SD_4k_1 = ssd.SparseDeconvolutionModel(*data_4k_1, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_4k_2 = ssd.SparseDeconvolutionModel(*data_4k_2, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)

# <codecell>

vol_anat = oio.get_t1(resample=ni.load(oio.data_path + '/%s/%s_wm_mask.nii.gz'%(subject, subject)))

# <codecell>

rrmse_data = [oza.cross_predict(SD_1k_1, SD_1k_2), oza.cross_predict(SD_2k_1, SD_2k_2), oza.cross_predict(SD_4k_1, SD_4k_2)]

# <codecell>

%gui wx
fn = []
for vol_rmse in rrmse_data:
    fn.append('%s.png'%tempfile.NamedTemporaryFile().name)
    viz.plot_cut_planes(vol_anat,
                        overlay=vol_rmse,
                        vmin=0.5,
                        vmax=1.5,
                        overlay_cmap="RdYlGn",
                        invert_cmap=True,
                        slice_coronal=40,
                        slice_saggital=15,
                        slice_axial=30,
                        view_azim=-40,
                        view_elev=60,
                        file_name=fn[-1])

# <codecell>

for this_fn in fn:
    i = Image(filename=this_fn, width=1280, height=1024)
    display(i)

# <codecell>

fn = []
for vol_rmse in rrmse_data:
    fn.append('%s.png'%tempfile.NamedTemporaryFile().name)
    viz.plot_cut_planes(vol_anat,
                        overlay=vol_rmse,
                        vmin=0.5,
                        vmax=1.5,
                        overlay_cmap="RdYlGn",
                        invert_cmap=True,
                        slice_coronal=40,
                        slice_saggital=15,
                        slice_axial=45,
                        view_azim=40,
                        view_elev=60,
                        file_name=fn[-1])

# <codecell>

for this_fn in fn:
    i = Image(filename=this_fn, width=1280, height=1024)
    display(i)

# <codecell>

viz.plot_cut_planes(vol_anat,
                    overlay=rrmse_data[-1],
                    vmin=0.5,
                    vmax=1.5,
                    overlay_cmap="RdYlGn",
                    invert_cmap=True,
                    slice_coronal=None,
                    slice_saggital=None,
                    slice_axial=45,
                    view_azim=0,
                    view_elev=0)

# <codecell>

%gui wx
viz.plot_cut_planes(vol_anat,
                    overlay=rrmse_data[-1],
                    vmin=0.5,
                    vmax=1.5,
                    overlay_cmap="RdYlGn",
                    invert_cmap=True,
                    slice_coronal=None,
                    slice_saggital=None,
                    slice_axial=30,
                    view_azim=0,
                    view_elev=0)

# <codecell>


