# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# This takes a (very long) while to run unless the tools in osmosis.parallel are 
# used to parallelize the computation of all the 

import nibabel as ni
import os

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

data_fnames = {1000:[data_1k_1, data_1k_2], 2000:[data_2k_1, data_2k_2], 4000:[data_4k_1, data_4k_2]}

# <codecell>

data_fnames

# <codecell>

# Read all the data into arrays up front to save time in the main loop below:
data = {}
for b in data_fnames:
    data[b] = [[ni.load(data_fnames[b][i][0]).get_data(), np.loadtxt(data_fnames[b][i][1]), np.loadtxt(data_fnames[b][i][2])] for i in [0,1]]

# <codecell>

wm_mask = np.zeros(ni.load(data_1k_1[0]).shape[:3])
wm_nifti = ni.load(oio.data_path + '/%s/%s_wm_mask.nii.gz'%(subject, subject)).get_data()
wm_mask[np.where(wm_nifti==1)] = 1

# <codecell>

alphas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
l1_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# <codecell>

def tensor_reliability(SD_1, SD_2, wm_mask):
    """ 
    given two SD models, calculate the reliability of the PDD of the derived tensor models
    """
    pred1 = np.concatenate([SD_1.S0[...,np.newaxis], SD_1.fit], -1)
    pred2 = np.concatenate([SD_2.S0[...,np.newaxis], SD_2.fit], -1)
    
    new_bvecs1 = np.concatenate([np.array([[0,0,0]]).T, SD_1.bvecs[:, SD_1.b_idx]], -1)
    new_bvecs2 = np.concatenate([np.array([[0,0,0]]).T, SD_2.bvecs[:, SD_2.b_idx]], -1)
    
    # Freely assume that the bvals are the same 
    new_bvals = np.hstack([0, SD_1.bvals[:,SD_1.b_idx]])
    TM1 = dti.TensorModel(pred1, new_bvecs1, new_bvals, mask=wm_mask, params_file='temp')
    TM2 = dti.TensorModel(pred2, new_bvecs2, new_bvals, mask=wm_mask, params_file='temp')
    pdd_rel = oza.pdd_reliability(TM1, TM2)
    
    return pdd_rel

# <codecell>

rmse_matrix = {}
rel_matrix = {}
for b in [1000, 2000, 4000]:
    ad_rd = oio.get_ad_rd(subject, b)
    rmse_matrix[b] = np.zeros((len(rhos), len(alphas)))
    rel_matrix[b] = np.zeros((len(rhos), len(alphas)))
    for l1_idx, l1_ratio in enumerate(l1_ratios):
        for alpha_idx, this_alpha in enumerate(alphas):
            solver_params = dict(alpha=this_alpha,
                                 l1_ratio=l1_ratio,
                                 fit_intercept=False,
                                 positive=True)
            
            params_file1 = "%s_SSD_l1ratio%s_alpha%s.nii.gz"%(oio.data_path + '/%s/'%subject + data_fnames[b][0][0].split('/')[-1].split('.')[0],l1_ratio, this_alpha)
            SD_1 = ssd.SparseDeconvolutionModel(*data[b][0], mask=wm_mask, params_file=params_file1, solver_params=solver_params, 
                                                axial_diffusivity=ad_rd[0]['AD'], radial_diffusivity=ad_rd[0]['RD'])
            params_file2 = "%s_SSD_l1ratio%s_alpha%s.nii.gz"%(oio.data_path + '/%s/'%subject + data_fnames[b][1][0].split('/')[-1].split('.')[0],l1_ratio, this_alpha)
            SD_2 = ssd.SparseDeconvolutionModel(*data[b][1], mask=wm_mask, params_file=params_file2, solver_params=solver_params,
                                                axial_diffusivity=ad_rd[1]['AD'], radial_diffusivity=ad_rd[1]['RD'])
            
            rrmse = oza.cross_predict(SD_1, SD_2)
            median_rmse = np.median(rrmse[np.isfinite(rrmse)])
            rmse_matrix[b][l1_idx, alpha_idx] = median_rmse
            
            rel = tensor_reliability(SD_1, SD_2, wm_mask)
            median_rel = np.median(rel[np.isfinite(rel)])
            rel_matrix[b][l1_idx, alpha_idx] = median_rel
            

# <codecell>

for b in [1000, 2000, 4000]:
    fig, ax = plt.subplots(1)
    #cax = ax.imshow(rmse_matrix[b], interpolation='nearest', cmap=cm.RdYlGn_r, vmax=1.5, vmin=0.5)
    #cbar = fig.colorbar(cax, ticks=[0.5, 1., 1.5])
    #cax.axes.set_xticks([0,1,2,3,4,5,6,7,8])
    #cax.axes.set_xticklabels([str('%1.4f'%this) for this in alphas])
    #cax.axes.set_xlabel(r'$\lambda$')
    #cax.axes.set_yticks([0,1,2,3,4,5])
    #cax.axes.set_yticklabels([str('%1.1f'%(1-this)) for this in rhos])

    cax.axes.set_ylabel(r'$\alpha$')
    im = ax.matshow(rmse_matrix[b], cmap=matplotlib.cm.RdYlGn_r)
    fig.set_size_inches([8,6])
    fig.savefig('/home/arokem/Dropbox/osmosis_paper_figures/Figure5_acc_b%s.svg'%b)
    plt.colorbar(im)
rmse_matrix[1000]

# <codecell>

#plot(rmse_matrix[1000][:,0])
#plot(rmse_matrix[1000][:,1])
#plot(rmse_matrix[1000][:,2])
#plot(rmse_matrix[1000][:,-1])

#print rmse_matrix[1000][4,0],rmse_matrix[1000][1,0]
#print rmse_matrix[2000][4,0],rmse_matrix[2000][1,0]
#print rmse_matrix[4000][4,0],rmse_matrix[4000][1,0]

#print rmse_matrix[1000][4,1],rmse_matrix[1000][1,1]
#print rmse_matrix[2000][4,1],rmse_matrix[2000][1,1]
#print rmse_matrix[4000][4,1],rmse_matrix[4000][1,1]

print rmse_matrix[1000][0,7],rmse_matrix[1000][2,1]
print rmse_matrix[2000][0,7],rmse_matrix[2000][2,1]
print rmse_matrix[4000][0,7],rmse_matrix[4000][2,1]


print rmse_matrix[1000][5,0],rmse_matrix[1000][2,1]
print rmse_matrix[2000][5,0],rmse_matrix[2000][2,1]
print rmse_matrix[4000][5,0],rmse_matrix[4000][2,1]


print np.where(rmse_matrix[1000]==np.min(rmse_matrix[1000]))
#print np.where(rmse_matrix[2000]==np.min(rmse_matrix[2000]))
#print np.where(rmse_matrix[4000]==np.min(rmse_matrix[4000]))

#print np.min(rmse_matrix[1000])
#print np.min(rmse_matrix[2000])
#print np.min(rmse_matrix[4000])

# <codecell>

for b in [1000, 2000, 4000]:
    fig, ax = plt.subplots(1)
    cax = ax.imshow(rel_matrix[b], interpolation='nearest', cmap=cm.OrRd, vmax=90, vmin=0)
    cbar = fig.colorbar(cax)
    cax.axes.set_xticks([0,1,2,3,4,5,6,7,8])
    cax.axes.set_xticklabels([str('%1.4f'%this) for this in alphas])
    cax.axes.set_xlabel(r'$\lambda$')
    cax.axes.set_yticks([0,1,2,3,4,5])
    cax.axes.set_yticklabels([str('%1.1f'%(1-this)) for this in rhos])

    cax.axes.set_ylabel(r'$\alpha$')
    #ax.matshow(rmse_matrix[b], cmap=matplotlib.cm.hot)
    fig.set_size_inches([8,6])
    fig.savefig('/home/arokem/Dropbox/osmosis_paper_figures/Figure5_rel_b%s.svg'%b)

# <codecell>

print rel_matrix[1000][4,0],rel_matrix[1000][1,2]
print rel_matrix[2000][4,0],rel_matrix[2000][1,2]
print rel_matrix[4000][4,0],rel_matrix[4000][1,2]

# <codecell>


