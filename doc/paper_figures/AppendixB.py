# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.stats as stats

import nibabel as ni

import osmosis.viz.mpl as viz
import osmosis.utils as ozu
import osmosis.model.sparse_deconvolution as ssd
import osmosis.model.dti as dti
import osmosis.model.analysis as oza

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

mask = np.zeros(ni.load(data_1k_1[0]).shape[:3])
wm_nifti = ni.load(oio.data_path + '/%s/%s_wm_mask.nii.gz'%(subject, subject)).get_data()
mask[np.where(wm_nifti==1)] = 1

# <codecell>

TM_1k_1 = dti.TensorModel(*data_1k_1, mask=mask, params_file='temp')
TM_1k_2 = dti.TensorModel(*data_1k_2, mask=mask, params_file='temp')
TM_2k_1 = dti.TensorModel(*data_2k_1, mask=mask, params_file='temp')
TM_2k_2 = dti.TensorModel(*data_2k_2, mask=mask, params_file='temp')
TM_4k_1 = dti.TensorModel(*data_4k_1, mask=mask, params_file='temp')
TM_4k_2 = dti.TensorModel(*data_4k_2, mask=mask, params_file='temp')

# <codecell>

rmse1k = oza.rmse(TM_1k_1, TM_1k_2)
rmse2k = oza.rmse(TM_2k_1, TM_2k_2)
rmse4k = oza.rmse(TM_4k_1, TM_4k_2)

rmse1k = rmse1k[np.isfinite(rmse1k)]
rmse2k = rmse2k[np.isfinite(rmse2k)]
rmse4k = rmse4k[np.isfinite(rmse4k)]

# <codecell>

fig = viz.probability_hist(rmse1k, label='b=1000')
fig = viz.probability_hist(rmse2k, fig=fig, label='b=2000')
fig = viz.probability_hist(rmse4k, fig=fig, label='b=4000')
ax = fig.axes[0]
ax.set_xlabel('RMSE')
ax.set_ylabel('P(RMSE)')
plt.legend()
ax.text(80, 0.05, 'median(1000): %2.2f \nmedian(2000): %2.2f \nmedian(4000): %2.2f'%(np.median(rmse1k), np.median(rmse2k), np.median(rmse4k)))
fig.savefig('/home/arokem/Dropbox/rmse_distributions.svg')

# <codecell>

fig, ax = plt.subplots(1)

rmse1k = rmse1k[np.isfinite(rmse1k)]
rmse2k = rmse2k[np.isfinite(rmse2k)]
rmse4k = rmse4k[np.isfinite(rmse4k)]

m = [np.median(x) for x in [rmse1k, rmse2k, rmse4k]]
e_up = [np.abs(np.median(x)-stats.scoreatpercentile(x,0.95)) for x in [rmse1k, rmse2k, rmse4k]]
e_down = [np.abs(np.median(x)-stats.scoreatpercentile(x,0.05)) for x in [rmse1k, rmse2k, rmse4k]]
            
ax.errorbar([1000,2000,4000], m, yerr=np.vstack([e_up, e_down]), 
            fmt = 'o')

ax.set_ylabel('RMSE')
ax.set_xlabel('b value')
ax.set_xlim([900, 4500])
ax.set_ylim([10,100])
ax.set_xticks([1000,2000,4000])
ax.loglog()
fig.savefig('/home/arokem/Dropbox/rmse_bars.svg')

# <codecell>

def calc_snr(model):
    b0 = model.data[..., model.b0_idx][model.mask]
    b_data = model.data[..., model.b_idx][model.mask]
    s = np.mean(b_data, -1)
    sigma = np.std(b0, -1)
    # Correct for small sample (according to http://nbviewer.ipython.org/4287207)
    nb0 = len(model.b0_idx)
    bias=sigma*(1-np.sqrt(2/(nb0-1))*(gamma(nb0/2)/gamma((nb0-1)/2)))
    n = sigma + bias
    return s/n

# <codecell>

snr1k = calc_snr(TM_1k_1)
snr2k = calc_snr(TM_2k_1)
snr4k = calc_snr(TM_4k_1)

# <codecell>

fig = viz.probability_hist(snr1k, label='b=1000')
fig = viz.probability_hist(snr2k, fig=fig, label='b=2000')
fig = viz.probability_hist(snr4k, fig=fig, label='b=4000')
ax = fig.axes[0]
ax.set_xlabel('SNR')
ax.set_ylabel('P(SNR)')
plt.legend()
ax.text(20, 0.20, 'median(1000): %2.2f \nmedian(2000): %2.2f \nmedian(4000): %2.2f'%(np.median(snr1k), np.median(snr2k), np.median(snr4k)))
fig.savefig('/home/arokem/Dropbox/snr_distributions.svg')

# <codecell>

fig, ax = plt.subplots(1)

m = [np.median(x) for x in [snr1k, snr2k, snr4k]]
e_up = [np.abs(np.median(x)-stats.scoreatpercentile(x,0.95)) for x in [snr1k, snr2k, snr4k]]
e_down = [np.abs(np.median(x)-stats.scoreatpercentile(x,0.05)) for x in [snr1k, snr2k, snr4k]]
            
ax.errorbar([1000,2000,4000], m, yerr=np.vstack([e_up, e_down]), 
            fmt = 'o')

ax.set_ylabel('SNR')
ax.set_xlabel('b value')
ax.set_xlim([900, 4500])
ax.set_ylim([0, 40])
ax.set_xticks([1000,2000,4000])


ax.loglog()
fig.savefig('/home/arokem/Dropbox/snr_bars.svg')

