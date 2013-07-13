# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.stats as stats

import nibabel as ni

import osmosis as oz
import osmosis.viz.mpl as viz
import osmosis.utils as ozu
import osmosis.io as oio
import osmosis.model.sparse_deconvolution as ssd
import osmosis.model.dti as dti
import osmosis.model.analysis as oza

oio.data_path = '/biac4/wandell/biac2/wandell6/data/arokem/osmosis'

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

# This is the best according to rRMSE across bvals: 
l1_ratio = 0.8
alpha = 0.0005 
solver_params = dict(l1_ratio=l1_ratio, alpha=alpha, fit_intercept=False, positive=True)

# <codecell>

ad_rd = oio.get_ad_rd(subject, 1000)
SD_1k_1 = ssd.SparseDeconvolutionModel(*data_1k_1, mask=mask, solver_params=solver_params, params_file = 'temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
SD_1k_2 = ssd.SparseDeconvolutionModel(*data_1k_2, mask=mask, solver_params=solver_params, params_file = 'temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])
ad_rd = oio.get_ad_rd(subject, 2000)
SD_2k_1 = ssd.SparseDeconvolutionModel(*data_2k_1, mask=mask, solver_params=solver_params, params_file = 'temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
SD_2k_2 = ssd.SparseDeconvolutionModel(*data_2k_2, mask=mask, solver_params=solver_params, params_file = 'temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])
ad_rd = oio.get_ad_rd(subject, 4000)
SD_4k_1 = ssd.SparseDeconvolutionModel(*data_4k_1, mask=mask, solver_params=solver_params, params_file = 'temp', axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
SD_4k_2 = ssd.SparseDeconvolutionModel(*data_4k_2, mask=mask, solver_params=solver_params, params_file = 'temp', axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])

# <codecell>

TM_1k_1 = dti.TensorModel(*data_1k_1, mask=mask, params_file='temp')
TM_1k_2 = dti.TensorModel(*data_1k_2, mask=mask, params_file='temp')
TM_2k_1 = dti.TensorModel(*data_2k_1, mask=mask, params_file='temp')
TM_2k_2 = dti.TensorModel(*data_2k_2, mask=mask, params_file='temp')
TM_4k_1 = dti.TensorModel(*data_4k_1, mask=mask, params_file='temp')
TM_4k_2 = dti.TensorModel(*data_4k_2, mask=mask, params_file='temp')

# <codecell>

dtm_rrmse_1k = oza.cross_predict(TM_1k_1, TM_1k_2) 
dtm_rrmse_2k = oza.cross_predict(TM_2k_1, TM_2k_2)
dtm_rrmse_4k = oza.cross_predict(TM_4k_1, TM_4k_2)

sfm_rrmse_1k = oza.cross_predict(SD_1k_1, SD_1k_2) 
sfm_rrmse_2k = oza.cross_predict(SD_2k_1, SD_2k_2)
sfm_rrmse_4k = oza.cross_predict(SD_4k_1, SD_4k_2)

# <codecell>

idx = np.logical_and(np.logical_and(np.isfinite(dtm_rrmse_1k), dtm_rrmse_1k<1.5), np.isfinite(sfm_rrmse_1k))
fig, cbar = viz.scatter_density(np.hstack([0.5, dtm_rrmse_1k[idx], 1.5]), np.hstack([0.5, sfm_rrmse_1k[idx], 1.5]), return_cbar=True, vmin=0, vmax=3)
fig.axes[0].plot([0,100],[100,0], 'k--')
fig.axes[0].plot([100,0],[50,50], 'k--')
fig.axes[0].plot([50,50],[100,0], 'k--')

cbar.set_ticks([0,1,2,3])
cbar.set_ticklabels([1,10,100,1000])

fig.set_size_inches([10,10])
#fig.savefig('/home/arokem/Dropbox/scatter_rrmse_1k.svg')
fig_hist = viz.probability_hist(dtm_rrmse_1k[idx] - sfm_rrmse_1k[idx])
fig_hist.set_size_inches([8,6])

# <codecell>

idx = np.logical_and(np.logical_and(np.isfinite(dtm_rrmse_2k), dtm_rrmse_2k<1.5), np.isfinite(sfm_rrmse_2k))
fig, cbar = viz.scatter_density(np.hstack([0.5, dtm_rrmse_2k[idx], 1.5]), np.hstack([0.5, sfm_rrmse_2k[idx], 1.5]), return_cbar=True, vmin=0, vmax=3)
fig.axes[0].plot([0,100],[100,0], 'k--')
fig.axes[0].plot([100,0],[50,50], 'k--')
fig.axes[0].plot([50,50],[100,0], 'k--')
cbar.set_ticks([0,1,2,3])
cbar.set_ticklabels([1,10,100,1000])

fig.set_size_inches([10,10])
#fig.savefig('/home/arokem/Dropbox/scatter_rrmse_2k.svg')

# <codecell>

idx = np.logical_and(np.logical_and(np.isfinite(dtm_rrmse_4k), dtm_rrmse_4k<1.5), np.isfinite(sfm_rrmse_4k))
fig, cbar = viz.scatter_density(np.hstack([0.5, dtm_rrmse_4k[idx], 1.5]), np.hstack([0.5, sfm_rrmse_4k[idx], 1.5]), return_cbar=True, vmin=0, vmax=3)
fig.axes[0].plot([0,100],[100,0], 'k--')
fig.axes[0].plot([100,0],[50,50], 'k--')
fig.axes[0].plot([50,50],[100,0], 'k--')
cbar.set_ticks([0,1,2,3])
cbar.set_ticklabels([1,10,100,1000])

fig.set_size_inches([10,10])
#fig.savefig('/home/arokem/Dropbox/scatter_rrmse_4k.svg')

# <codecell>

def permutation_test_rel(arr1, arr2, n_perms=1000):
    """
    Test the difference between two nd arrays

    """
    # Linearize and get the finite elements: 
    arr1 = arr1[np.isfinite(arr1)]
    arr2 = arr2[np.isfinite(arr2)]  # We assume that the indices are good for both
    orig_arr = np.vstack([arr1, arr2])
    orig_diff = np.median(arr1 - arr2)
    perm_diff = np.empty(1000)
    for ii in xrange(n_perms):
        idx1 = np.random.randint(0, 2, orig_arr.shape[-1])
        idx2 = np.mod(idx1+1, 2)
        perm_arr1 = orig_arr[idx1, np.arange(idx1.shape[0])] 
        perm_arr2 = orig_arr[idx2, np.arange(idx2.shape[0])]
        perm_diff[ii] = np.median(perm_arr1 - perm_arr2)
        
    return orig_diff, perm_diff

# <codecell>

orig_diff1k, medians1k = permutation_test_rel(dtm_rrmse_1k, sfm_rrmse_1k)
orig_diff2k, medians2k = permutation_test_rel(dtm_rrmse_2k, sfm_rrmse_2k)
orig_diff4k, medians4k = permutation_test_rel(dtm_rrmse_4k, sfm_rrmse_4k)

# <codecell>

ci_1k = stats.scoreatpercentile(medians1k, 97.5) - stats.scoreatpercentile(medians1k, 2.5) 
ci_2k = stats.scoreatpercentile(medians2k, 97.5) - stats.scoreatpercentile(medians2k, 2.5) 
ci_4k = stats.scoreatpercentile(medians4k, 97.5) - stats.scoreatpercentile(medians4k, 2.5) 

# <codecell>

def boots_strap_medians(arr, n_iters=1000): 
    """ 
    Produce a boot-strap distribution of the median of an array
    """ 
    # Linearize and exclude nan/infs:
    arr = arr[np.isfinite(arr)]
    medians = np.empty(n_iters)
    for ii in xrange(n_iters): 
        this_arr = arr[np.random.random_integers(0, arr.shape[0]-1, arr.shape[0])]
        medians[ii] = np.median(this_arr)
    
    return medians

# <codecell>

medians_dtm_1k = boots_strap_medians(dtm_rrmse_1k)
medians_dtm_2k = boots_strap_medians(dtm_rrmse_2k)
medians_dtm_4k = boots_strap_medians(dtm_rrmse_4k)

medians_sfm_1k = boots_strap_medians(sfm_rrmse_1k)
medians_sfm_2k = boots_strap_medians(sfm_rrmse_2k)
medians_sfm_4k = boots_strap_medians(sfm_rrmse_4k)

# <codecell>

dtm_ci_1k = (stats.scoreatpercentile(medians_dtm_1k, 97.5) - stats.scoreatpercentile(medians_dtm_1k, 2.5))
sfm_ci_1k = (stats.scoreatpercentile(medians_sfm_1k, 97.5) - stats.scoreatpercentile(medians_sfm_1k, 2.5))

dtm_ci_2k = (stats.scoreatpercentile(medians_dtm_2k, 97.5) - stats.scoreatpercentile(medians_dtm_2k, 2.5))
sfm_ci_2k = (stats.scoreatpercentile(medians_sfm_2k, 97.5) - stats.scoreatpercentile(medians_sfm_2k, 2.5))

dtm_ci_4k = (stats.scoreatpercentile(medians_dtm_4k, 97.5) - stats.scoreatpercentile(medians_dtm_4k, 2.5))
sfm_ci_4k = (stats.scoreatpercentile(medians_sfm_4k, 97.5) - stats.scoreatpercentile(medians_sfm_4k, 2.5))

# <codecell>

fig, ax = plt.subplots(1)
ax.bar([1,2], 
       [np.median(dtm_rrmse_1k[np.isfinite(dtm_rrmse_1k)]), 
        np.median(sfm_rrmse_1k[np.isfinite(sfm_rrmse_1k)])],
        yerr=[dtm_ci_1k, sfm_ci_1k], ecolor='k')

ax.bar([4,5], 
        [np.median(dtm_rrmse_2k[np.isfinite(dtm_rrmse_2k)]), 
        np.median(sfm_rrmse_2k[np.isfinite(sfm_rrmse_2k)])], 
        yerr=[dtm_ci_2k, sfm_ci_2k],ecolor='k',
        color='g')

ax.bar([7,8], 
        [np.median(dtm_rrmse_4k[np.isfinite(dtm_rrmse_4k)]), 
        np.median(sfm_rrmse_4k[np.isfinite(sfm_rrmse_4k)])], 
        yerr=[dtm_ci_4k, sfm_ci_4k], ecolor='k',
        color='r')
ax.set_ylim([1/np.sqrt(2), 0.8])
ax.set_xlim([0.8, 9])
ax.grid()
ax.set_xticks([0])
fig.set_size_inches([10,6])
fig.savefig('/home/arokem/Dropbox/osmosis_paper_figures/rrmse_bars.svg')

# <codecell>

print [np.median(dtm_rrmse_1k[np.isfinite(dtm_rrmse_1k)]), 
        np.median(sfm_rrmse_1k[np.isfinite(sfm_rrmse_1k)])]
print [dtm_ci_1k, sfm_ci_1k]

print [np.median(dtm_rrmse_2k[np.isfinite(dtm_rrmse_2k)]), 
        np.median(sfm_rrmse_2k[np.isfinite(sfm_rrmse_2k)])]
print [dtm_ci_2k, sfm_ci_2k]

print [np.median(dtm_rrmse_4k[np.isfinite(dtm_rrmse_4k)]), 
        np.median(sfm_rrmse_4k[np.isfinite(sfm_rrmse_4k)])]
print [dtm_ci_1k, sfm_ci_1k]

