# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import nibabel as ni
import osmosis.model.analysis as oza
import osmosis.model.dti as dti
import osmosis.viz.mpl as mpl

# <codecell>

import os
import osmosis as oz
import osmosis.io as oio
oio.data_path = os.path.join(oz.__path__[0], 'data')

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

TM_1k_1 = dti.TensorModel(*data_1k_1, mask=wm_mask, params_file='temp')
TM_1k_2 = dti.TensorModel(*data_1k_2, mask=wm_mask, params_file='temp')
TM_2k_1 = dti.TensorModel(*data_2k_1, mask=wm_mask, params_file='temp')
TM_2k_2 = dti.TensorModel(*data_2k_2, mask=wm_mask, params_file='temp')
TM_4k_1 = dti.TensorModel(*data_4k_1, mask=wm_mask, params_file='temp')
TM_4k_2 = dti.TensorModel(*data_4k_2, mask=wm_mask, params_file='temp')

# <codecell>

rrmse_1k = oza.cross_predict(TM_1k_1, TM_1k_2)
rrmse_2k = oza.cross_predict(TM_2k_1, TM_2k_2)
rrmse_4k = oza.cross_predict(TM_4k_1, TM_4k_2)

# <codecell>

fig = mpl.probability_hist(rrmse_1k[np.isfinite(rrmse_1k)], label='b=%s'%str(1000))
fig = mpl.probability_hist(rrmse_2k[np.isfinite(rrmse_2k)], fig=fig, label='b=%s'%str(2000))
fig = mpl.probability_hist(rrmse_4k[np.isfinite(rrmse_4k)], fig=fig, label='b=%s'%str(4000))
ax = fig.axes[0]
ax.set_xlim([0.6, 1.4])
fig.axes[0].plot([1,1], [ax.get_ylim()[0], ax.get_ylim()[1]], '--k')
fig.axes[0].plot([1/np.sqrt(2),1/np.sqrt(2)], [ax.get_ylim()[0], ax.get_ylim()[1]], '--k')

plt.legend()
fig.savefig('figures/Figure2_histogram.svg')

# <codecell>

for this in [rrmse_1k,rrmse_2k, rrmse_4k]:
    isfin = this[np.isfinite(this)]
    print "The proportion of voxels with rRMSE<1.0 is %s"%(100 * len(np.where(isfin<1)[0])/float(len(isfin)))

# <codecell>

# Look at the same corpus callosum voxel as in Figure 1: 
vox_idx = (40, 73, 32)  # This doesn't make sense in HT's brain.

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

fig.savefig('figures/Figure2_scatter_self.svg')

# <codecell>

fig, axes = plt.subplots(1,3,squeeze=True)
for ax, models in zip(axes,([TM_1k_1,TM_1k_2],[TM_2k_1, TM_2k_2],[TM_4k_1, TM_4k_2])):
    ax.scatter(models[0].relative_signal[vox_idx], models[1].fit[vox_idx]/models[1].S0[vox_idx])
    ax.grid()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')
    ax.set_xlabel(r'Measured $\frac{S}{S_0}$')
axes[0].set_ylabel(r'Predicted $\frac{S}{S_0}$')    
fig.set_size_inches([10,10])

fig.savefig('figures/Figure2_scatter_cross.svg')
print rrmse_1k[vox_idx],rrmse_2k[vox_idx], rrmse_4k[vox_idx]

