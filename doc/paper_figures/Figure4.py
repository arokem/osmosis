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

# This is the best according to rRMSE across bvals: 
l1_ratio = 0.8
alpha = 0.0005 
solver_params = dict(l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False, positive=True)

# <codecell>

#ad_rd = oio.get_ad_rd(subject, 1000)
#SD_1k_1 = ssd.SparseDeconvolutionModel(*data_1k_1, mask=wm_mask,  params_file = data_1k_1[0].split('.')[0] + params_string, axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
#SD_1k_2 = ssd.SparseDeconvolutionModel(*data_1k_2, mask=wm_mask,  params_file = data_1k_2[0].split('.')[0] + params_string, axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
#ad_rd = oio.get_ad_rd(subject, 2000)
#SD_2k_1 = ssd.SparseDeconvolutionModel(*data_2k_1, mask=wm_mask,  params_file = data_2k_1[0].split('.')[0] + params_string, axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
#SD_2k_2 = ssd.SparseDeconvolutionModel(*data_2k_2, mask=wm_mask,  params_file = data_2k_2[0].split('.')[0] + params_string, axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
#ad_rd = oio.get_ad_rd(subject, 4000)
#SD_4k_1 = ssd.SparseDeconvolutionModel(*data_4k_1, mask=wm_mask,  params_file = data_4k_1[0].split('.')[0] + params_string, axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
#SD_4k_2 = ssd.SparseDeconvolutionModel(*data_4k_2, mask=wm_mask,  params_file = data_4k_2[0].split('.')[0] + params_string, axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)

#TM_4k_1 = dti.TensorModel(*data_4k_1, mask=wm_mask)
#TM_4k_2 = dti.TensorModel(*data_4k_2, mask=wm_mask)

ad_rd = oio.get_ad_rd(subject, 1000)
SD_1k_1 = ssd.SparseDeconvolutionModel(*data_1k_1, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_1k_2 = ssd.SparseDeconvolutionModel(*data_1k_2, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
ad_rd = oio.get_ad_rd(subject, 2000)
SD_2k_1 = ssd.SparseDeconvolutionModel(*data_2k_1, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_2k_2 = ssd.SparseDeconvolutionModel(*data_2k_2, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)
ad_rd = oio.get_ad_rd(subject, 4000)
SD_4k_1 = ssd.SparseDeconvolutionModel(*data_4k_1, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'], solver_params=solver_params)
SD_4k_2 = ssd.SparseDeconvolutionModel(*data_4k_2, mask=wm_mask,  params_file ='temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'], solver_params=solver_params)

#TM_4k_1 = dti.TensorModel(*data_4k_1, mask=wm_mask)
#TM_4k_2 = dti.TensorModel(*data_4k_2, mask=wm_mask)

# <codecell>

rrmse_1k = oza.cross_predict(SD_1k_1, SD_1k_2)
rrmse_2k = oza.cross_predict(SD_2k_1, SD_2k_2)
rrmse_4k = oza.cross_predict(SD_4k_1, SD_4k_2)
#rrmse_tensor_4k = oza.cross_predict(TM_4k_1, TM_4k_2)

# <codecell>

fig = mpl.probability_hist(rrmse_1k[np.isfinite(rrmse_1k)], label='b=%s'%str(1000))
fig = mpl.probability_hist(rrmse_2k[np.isfinite(rrmse_2k)], fig=fig, label='b=%s'%str(2000))
fig = mpl.probability_hist(rrmse_4k[np.isfinite(rrmse_4k)], fig=fig, label='b=%s'%str(4000))
# Add one of the tensor curves from Figure 2 and put it in the background as reference:    
# fig = mpl.probability_hist(rrmse_tensor_4k[np.isfinite(rrmse_4k)], fig=fig, color='gray', label='Tensor model at b=%s'%str(4000))
#fig.set_size_inches([10, 8])
fig.axes[0].plot([1,1], fig.axes[0].get_ylim(), '--k')
fig.axes[0].plot([1/np.sqrt(2),1/np.sqrt(2)], fig.axes[0].get_ylim(), '--k')
fig.axes[0].set_xlim([0.6,1.4])
plt.legend()

fig.savefig('/home/arokem/Dropbox/osmosis_paper_figures/Figure4_histogram.svg')

# <codecell>

for this in [rrmse_1k, rrmse_2k, rrmse_4k]:
    isfin = this[np.isfinite(this)]
    print "The proportion of voxels with rRMSE<1.0 is %s"%(100 * len(np.where(isfin<1)[0])/float(len(isfin)))
        

# <codecell>


