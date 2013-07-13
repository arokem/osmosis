# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import nibabel as ni

import osmosis as oz
import osmosis.model.analysis as oza
import osmosis.model.sparse_deconvolution as ssd
import osmosis.model.dti as dti
import osmosis.viz.mpl as mpl

import osmosis.io as oio
oio.data_path = '/biac4/wandell/biac2/wandell6/data/arokem/osmosis'

# <codecell>

subject = 'FP'
data_1k_1, data_1k_2 = oio.get_dwi_data(1000, subject)
data_2k_1, data_2k_2 = oio.get_dwi_data(2000, subject)
data_4k_1, data_4k_2 = oio.get_dwi_data(4000, subject)

# <codecell>

wm_mask = np.zeros(ni.load(data_1k_1[0]).shape[:3])
wm_nifti = ni.load(oio.data_path + '/%s/%s_wm_mask.nii.gz'%(subject, subject)).get_data()
wm_mask[np.where(wm_nifti==1)] = 1

# <codecell>

l1_ratio = 0.8
alpha = 0.0005 
solver_params = dict(l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False, positive=True)

# <codecell>

ad_rd = oio.get_ad_rd(subject, 1000)
SD_1k_1 = ssd.SparseDeconvolutionModel(*data_1k_1, mask=wm_mask,  params_file = 'temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_1k_2 = ssd.SparseDeconvolutionModel(*data_1k_2, mask=wm_mask,  params_file = 'temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
ad_rd = oio.get_ad_rd(subject, 2000)
SD_2k_1 = ssd.SparseDeconvolutionModel(*data_2k_1, mask=wm_mask,  params_file = 'temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_2k_2 = ssd.SparseDeconvolutionModel(*data_2k_2, mask=wm_mask,  params_file = 'temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
ad_rd = oio.get_ad_rd(subject, 4000)
SD_4k_1 = ssd.SparseDeconvolutionModel(*data_4k_1, mask=wm_mask,  params_file = 'temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_4k_2 = ssd.SparseDeconvolutionModel(*data_4k_2, mask=wm_mask,  params_file = 'temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)

# <codecell>

pred_1k_1 = np.concatenate([SD_1k_1.S0[...,np.newaxis], SD_1k_1.fit], -1)
pred_1k_2 = np.concatenate([SD_1k_2.S0[...,np.newaxis], SD_1k_2.fit], -1)
new_bvecs1 = np.concatenate([np.array([[0,0,0]]).T, SD_1k_1.bvecs[:, SD_1k_1.b_idx]], -1)
new_bvecs2 = np.concatenate([np.array([[0,0,0]]).T, SD_1k_2.bvecs[:, SD_1k_2.b_idx]], -1)
new_bvals = np.hstack([0, SD_1k_1.bvals[:,SD_1k_1.b_idx]])
TM_1k_1 = dti.TensorModel(pred_1k_1, new_bvecs1, new_bvals, mask=wm_mask, params_file='temp')
TM_1k_2 = dti.TensorModel(pred_1k_2, new_bvecs2, new_bvals, mask=wm_mask, params_file='temp')

pred_2k_1 = np.concatenate([SD_2k_1.S0[...,np.newaxis], SD_2k_1.fit], -1)
pred_2k_2 = np.concatenate([SD_2k_2.S0[...,np.newaxis], SD_2k_2.fit], -1)
new_bvecs1 = np.concatenate([np.array([[0,0,0]]).T, SD_2k_1.bvecs[:, SD_2k_1.b_idx]], -1)
new_bvecs2 = np.concatenate([np.array([[0,0,0]]).T, SD_2k_2.bvecs[:, SD_2k_2.b_idx]], -1)
new_bvals = np.hstack([0, SD_2k_1.bvals[:,SD_2k_1.b_idx]])
TM_2k_1 = dti.TensorModel(pred_2k_1, new_bvecs1, new_bvals, mask=wm_mask, params_file='temp')
TM_2k_2 = dti.TensorModel(pred_2k_2, new_bvecs2, new_bvals, mask=wm_mask, params_file='temp')

pred_4k_1 = np.concatenate([SD_4k_1.S0[...,np.newaxis], SD_4k_1.fit], -1)
pred_4k_2 = np.concatenate([SD_4k_2.S0[...,np.newaxis], SD_4k_2.fit], -1)
new_bvecs1 = np.concatenate([np.array([[0,0,0]]).T, SD_4k_1.bvecs[:, SD_4k_1.b_idx]], -1)
new_bvecs2 = np.concatenate([np.array([[0,0,0]]).T, SD_4k_2.bvecs[:, SD_4k_2.b_idx]], -1)
new_bvals = np.hstack([0, SD_4k_1.bvals[:,SD_4k_1.b_idx]])
TM_4k_1 = dti.TensorModel(pred_4k_1, new_bvecs1, new_bvals, mask=wm_mask, params_file='temp')
TM_4k_2 = dti.TensorModel(pred_4k_2, new_bvecs2, new_bvals, mask=wm_mask, params_file='temp')

# <codecell>

# Reliability estimated via the tensor model: 
pdd_rel_1k = oza.pdd_reliability(TM_1k_1, TM_1k_2)
pdd_rel_2k = oza.pdd_reliability(TM_2k_1, TM_2k_2)
pdd_rel_4k = oza.pdd_reliability(TM_4k_1, TM_4k_2)

# Or directly on the SD model:
#pdd_rel_1k = oza.pdd_reliability(SD_1k_1, SD_1k_2)
#pdd_rel_2k = oza.pdd_reliability(SD_2k_1, SD_2k_2)
#pdd_rel_4k = oza.pdd_reliability(SD_4k_1, SD_4k_2)

# <codecell>

fig = mpl.probability_hist(pdd_rel_1k[np.isfinite(pdd_rel_1k)], label='b=%s'%str(1000))
print("For b=1000, the median PDD reliability is %2.2f"%np.median(pdd_rel_1k[np.isfinite(pdd_rel_1k)]))
fig = mpl.probability_hist(pdd_rel_2k[np.isfinite(pdd_rel_2k)], fig=fig, label='b=%s'%str(2000))
print("For b=2000, the median PDD reliability is %2.2f"%np.median(pdd_rel_2k[np.isfinite(pdd_rel_2k)]))
fig = mpl.probability_hist(pdd_rel_4k[np.isfinite(pdd_rel_4k)], fig=fig, label='b=%s'%str(4000))
print("For b=4000, the median PDD reliability is %2.2f"%np.median(pdd_rel_4k[np.isfinite(pdd_rel_4k)]))
ax = fig.axes[0]
ax.set_xlim([0,90])
plt.legend()
fig.savefig('/home/arokem/Dropbox/osmosis_paper_figures/Figure8_pdd_reliability.svg')

# <codecell>

pdd_rel_1k = oza.pdd_reliability(SD_1k_1, SD_1k_2)
pdd_rel_2k = oza.pdd_reliability(SD_2k_1, SD_2k_2)
pdd_rel_4k = oza.pdd_reliability(SD_4k_1, SD_4k_2)

# <codecell>

fig = mpl.probability_hist(pdd_rel_1k[np.isfinite(pdd_rel_1k)], label='b=%s'%str(1000))
print("For b=1000, the median PDD reliability is %2.2f"%np.median(pdd_rel_1k[np.isfinite(pdd_rel_1k)]))
fig = mpl.probability_hist(pdd_rel_2k[np.isfinite(pdd_rel_2k)], fig=fig, label='b=%s'%str(2000))
print("For b=2000, the median PDD reliability is %2.2f"%np.median(pdd_rel_2k[np.isfinite(pdd_rel_2k)]))
fig = mpl.probability_hist(pdd_rel_4k[np.isfinite(pdd_rel_4k)], fig=fig, label='b=%s'%str(4000))
print("For b=4000, the median PDD reliability is %2.2f"%np.median(pdd_rel_4k[np.isfinite(pdd_rel_4k)]))
ax = fig.axes[0]
ax.set_xlim([0,90])
plt.legend()
fig.savefig('/home/arokem/Dropbox/osmosis_paper_figures/sfm_pdd_reliability.svg')

# <codecell>


