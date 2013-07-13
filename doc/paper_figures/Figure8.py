# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import tempfile

import scipy.stats as stats

import dipy.core.geometry as geo
import dipy.sims.phantom as dps
import dipy.core.sphere as sphere
import osmosis.model.base as dwi
import osmosis.model.dti as dti
import osmosis.model.sparse_deconvolution as ssd
import osmosis.model.analysis as oza

import osmosis.utils as ozu
import osmosis.viz.maya as maya
import osmosis.viz.mpl as mpl
import osmosis.simulation as sim

import osmosis.tensor as ozt

# <codecell>

import os
import osmosis as oz
import osmosis.io as oio
oio.data_path = os.path.join(oz.__path__[0], 'data')

# <codecell>

n_sims = 500
subject = 'FP'

bvecs = []
bvals = []
data = []
b_idx = []

#bvals_to_test = [500, 800, 1000, 1200, 1500, 1800, 2000]#, 2500, 3000, 3500, 4000, 4500]
bvals_to_test = [1000, 2000, 4000]

# Always use 1000 data for the noise and S0:
data1, data2 =  oio.get_dwi_data(1000, subject)

for bval in bvals_to_test:
    bvals.append(np.loadtxt(data1[2])/1000. * bval)     
    bvecs.append(np.loadtxt(data1[1]))
    b_idx.append(np.where(bvals[-1]>0)[0])

mask = oio.get_wm_mask(oio.data_path + '/%s/%s_wm_mask.nii.gz'%(subject, subject), resample=data1[0])

# <codecell>

# We use the b=1000 to estimate the noise - it probably doesn't matter becasue it's all the same anyway:
DWI1 = dwi.DWI(*data1, mask=mask)
DWI2 = dwi.DWI(*data2, mask=mask)
noise_sig = (DWI1.data[DWI1.mask]- DWI2.data[DWI2.mask])/2.0
fig = mpl.probability_hist(ozu.rms(noise_sig))
S0 = [np.mean([stats.scoreatpercentile(DWI1._flat_S0, 25), stats.scoreatpercentile(DWI2._flat_S0, 25)]), 
      np.mean([np.median(DWI1._flat_S0), np.median(DWI2._flat_S0)]), 
      np.mean([stats.scoreatpercentile(DWI1._flat_S0, 75), stats.scoreatpercentile(DWI2._flat_S0, 75)])]

# <codecell>

# Likewise here, we take the first fascicle to point at one of the vectors from the b=4000 measurement:
vec1 = bvecs[-1][:, b_idx[-1][0]]
R = geo.rodriguez_axis_rotation(ozu.null_space([vec1, -vec1, [0,0,0]])[:,0], 90)
vec2 = np.array(np.dot(R.T, vec1)).squeeze()
n_bvecs=150

# <codecell>

angles = np.linspace(10, 90, 11)
angles = np.linspace(0,90,5)
weights = np.linspace(0.5, 0.5, 1)
sphere = np.array(S0) #np.linspace(S0,S0,1)
vol = []
vecs2keep = [] 

# <codecell>

weights

# <codecell>

vec1, vec2

# <codecell>

#deg_45 = np.pi/4
#n_vecs = 15
#vecs = []
#odf_weights = np.random.rand(n_vecs)
#odf_weights/= np.sum(odf_weights)
for bval_idx in range(len(bvals)):
    vol.append(np.empty((angles.shape[0], weights.shape[0], sphere.shape[0], n_bvecs + 10)))
    vecs2keep.append(np.empty((angles.shape[0], weights.shape[0], sphere.shape[0], 2, 3)))

    for i, angle in enumerate(angles):
        if angle==0:
            odf_vecs = np.array([vec1, vec1])
        else:
            # Rotate around the null-space of the vector and its inverse: 
            R = geo.rodriguez_axis_rotation(ozu.null_space([vec1, -vec1, [0,0,0]])[:,0], angle)
            vec2 = np.array(np.dot(R.T, vec1)).squeeze()
            odf_vecs = np.array([vec1, vec2])
        for j, w in enumerate(weights):
            for k, sph in enumerate(sphere):
                odf = sim.ODF(odf_vecs, [w, 1-w])
                vecs2keep[-1][i,j,k] = odf_vecs

                v = sim.Voxel(bvecs[bval_idx][:, 10:], bvals[bval_idx][10:], odf)
                S0 = 0 + np.ones(10) * sph
                vol[-1][i,j,k] = np.concatenate((S0, v.signal(np.mean(S0)))) 

# <codecell>

sfm_pdd_rel = []
sfm_pdd_acc = []
dtm_pdd_rel = []
dtm_pdd_acc = []


alpha = 0.0005
l1_ratio = 0.8
solver_params = dict(alpha=alpha,
                     l1_ratio=l1_ratio,
                     fit_intercept=False,
                     positive=True)

TM2keep = []
SFM2keep = []

for bval_idx in range(len(bvals)):
    
    this_vol = vol[bval_idx]
    this_vecs = vecs2keep[bval_idx]
    
    sfm_pdd_rel.append(np.zeros(vol[bval_idx].shape[:3] + (n_sims,)))
    dtm_pdd_rel.append(np.zeros(vol[bval_idx].shape[:3] + (n_sims,)))
    sfm_pdd_acc.append(np.zeros(vol[bval_idx].shape[:3] + (n_sims,)))
    dtm_pdd_acc.append(np.zeros(vol[bval_idx].shape[:3] + (n_sims,)))
    
    for sim_idx in xrange(n_sims): 
    
        noise_idx1 = int(np.random.rand() * noise_sig.shape[0])
        vol_noise1 = np.abs(this_vol + noise_sig[noise_idx1]) #abs is important - remember that the signal cannot go negative!
        
        noise_idx2 = int(np.random.rand() * noise_sig.shape[0])
        # Just making sure the two noise samples are actually different:
        while noise_idx1 == noise_idx2:
            noise_idx2 = int(np.random.rand() * sig_diff.shape[0])

        vol_noise2 = np.abs(this_vol + noise_sig[noise_idx2])
        
    
        TM1 = dti.TensorModel(vol_noise1,
                         bvecs[bval_idx],
                         bvals[bval_idx],
                         params_file = 'temp')

        TM2 = dti.TensorModel(vol_noise2,
                         bvecs[bval_idx],
                         bvals[bval_idx],
                         params_file = 'temp')

        SFM1 = ssd.SparseDeconvolutionModel(vol_noise1,
                                        bvecs[bval_idx],
                                        bvals[bval_idx],
                                        solver_params=solver_params,
                                        axial_diffusivity=1.5,
                                        radial_diffusivity=0.5,
                                        params_file = 'temp')

        SFM2 = ssd.SparseDeconvolutionModel(vol_noise2,
                                        bvecs[bval_idx],
                                        bvals[bval_idx],
                                        solver_params=solver_params,
                                        axial_diffusivity=1.5,
                                        radial_diffusivity=0.5,
                                        params_file = 'temp')
    
        
        for i, angle in enumerate(angles):
            for j, w in enumerate(weights):
                for k, sph in enumerate(sphere):
                    # Calculate the accuracy:
                    real_vec = vecs2keep[bval_idx][i,j,k]
                    model_vec = SFM1.principal_diffusion_direction[i,j,k]
                    model_vec = model_vec[np.unique(np.where(~np.isnan(model_vec))[0])]
                    if model_vec.shape[0] == 0:
                        sfm_pdd_acc[bval_idx][i,j,k, sim_idx] = np.mean(np.min(this_angs, 0)) 
                    else:
                        cross_vecs = np.dot(model_vec, real_vec.T)
                        cross_vecs[cross_vecs>1] = 1
                        this_angs = np.rad2deg(np.arccos(cross_vecs))
                        this_angs[this_angs>90] = 180 - this_angs[this_angs>90]
                        sfm_pdd_acc[bval_idx][i,j,k, sim_idx] = np.mean(np.min(this_angs, 0)) 
                        
                    dt_vec = TM1.principal_diffusion_direction[i,j,k]
                    dt_cross = np.dot(dt_vec, real_vec.T)
                    dt_cross[dt_cross>1] = 1
                    dt_angs = np.rad2deg(np.arccos(dt_cross))
                    dt_angs[dt_angs>90] = 180 - dt_angs[dt_angs>90]
                    dtm_pdd_acc[bval_idx][i,j,k, sim_idx] = np.mean(np.min(dt_angs))
                    
                    # Calculate the reliability:
                    this_sfm_ang = np.rad2deg(ozu.vector_angle(SFM1.principal_diffusion_direction[i,j,k][0], 
                                                               SFM2.principal_diffusion_direction[i,j,k][0]))
                    
                    sfm_pdd_rel[bval_idx][i,j,k,sim_idx] = np.min([this_sfm_ang, 180-this_sfm_ang])
                
                    this_dt_ang = np.rad2deg(ozu.vector_angle(TM1.principal_diffusion_direction[i,j,k], 
                                                          TM2.principal_diffusion_direction[i,j,k]))
                
                    dtm_pdd_rel[bval_idx][i,j,k,sim_idx] = np.min([this_dt_ang, 180-this_dt_ang])
                
    
    TM2keep.append(TM1)
    SFM2keep.append(SFM1)

# <codecell>

median_rel_sfm = []
median_rel_dtm = []
median_acc_sfm = []
median_acc_dtm = []

median_rel_sfm_e = []
median_rel_dtm_e = []
median_acc_sfm_e = []
median_acc_dtm_e = []

for bval_idx in range(len(bvals)):
    median_rel_sfm.append(np.zeros(sfm_pdd_rel[bval_idx].shape[:2]))
    median_rel_dtm.append(np.zeros(dtm_pdd_rel[bval_idx].shape[:2]))
    median_acc_sfm.append(np.zeros(sfm_pdd_acc[bval_idx].shape[:2]))
    median_acc_dtm.append(np.zeros(dtm_pdd_acc[bval_idx].shape[:2]))
    
    median_rel_sfm_e.append(np.zeros(sfm_pdd_rel[bval_idx].shape[:2]))
    median_rel_dtm_e.append(np.zeros(dtm_pdd_rel[bval_idx].shape[:2]))
    median_acc_sfm_e.append(np.zeros(sfm_pdd_acc[bval_idx].shape[:2]))
    median_acc_dtm_e.append(np.zeros(dtm_pdd_acc[bval_idx].shape[:2]))

    for i, angle in enumerate(angles):
            for j, w in enumerate(weights):
                for k, sph in enumerate(sphere):
                    median_rel_sfm[-1][i,j] = stats.nanmedian(sfm_pdd_rel[bval_idx][i,j,k,:])
                    median_rel_dtm[-1][i,j] = stats.nanmedian(dtm_pdd_rel[bval_idx][i,j,k,:])
                    median_acc_sfm[-1][i,j] = stats.nanmedian(sfm_pdd_acc[bval_idx][i,j,k,:])
                    median_acc_dtm[-1][i,j] = stats.nanmedian(dtm_pdd_acc[bval_idx][i,j,k,:])
                    
                    median_rel_sfm_e[-1][i,j] = stats.scoreatpercentile(sfm_pdd_rel[bval_idx][i,j,k,:], 84) - stats.scoreatpercentile(sfm_pdd_rel[bval_idx][i,j,k,:], 16) 
                    median_rel_dtm_e[-1][i,j] = stats.scoreatpercentile(dtm_pdd_rel[bval_idx][i,j,k,:], 84) - stats.scoreatpercentile(dtm_pdd_rel[bval_idx][i,j,k,:], 16)
                    median_acc_sfm_e[-1][i,j] = stats.scoreatpercentile(sfm_pdd_acc[bval_idx][i,j,k,:], 84) - stats.scoreatpercentile(sfm_pdd_acc[bval_idx][i,j,k,:], 16)                    
                    median_acc_dtm_e[-1][i,j] = stats.scoreatpercentile(dtm_pdd_acc[bval_idx][i,j,k,:], 84) - stats.scoreatpercentile(dtm_pdd_acc[bval_idx][i,j,k,:], 16)

median_rel_sfm = np.array(median_rel_sfm)
median_rel_dtm = np.array(median_rel_dtm)
median_acc_sfm = np.array(median_acc_sfm)
median_acc_dtm = np.array(median_acc_dtm)

median_rel_sfm_e = np.array(median_rel_sfm_e)
median_rel_dtm_e = np.array(median_rel_dtm_e)
median_acc_sfm_e = np.array(median_acc_sfm_e)
median_acc_dtm_e = np.array(median_acc_dtm_e)

# <codecell>

for yy, w_bal in enumerate(weights):
    fig = plt.figure()
    for ang_idx, ang in enumerate(angles):
        for bval_idx, bval in enumerate(bvals):
            lw = 3
            ax  = fig.add_subplot(len(bvals), len(angles), bval_idx * (len(angles)) + ang_idx+1, polar=True)
            ax.plot([0,0],[0,1], linewidth=lw, color='k')
            ax.plot([np.deg2rad(ang),np.deg2rad(ang)], [0,1], 'k', linewidth=lw)
            ax.set_yticklabels('')
            ax.set_yticks([])
            ax.set_xticklabels('')
            ax.bar(np.deg2rad(median_acc_dtm[bval_idx,ang_idx,yy])-np.deg2rad(median_rel_dtm[bval_idx,ang_idx,yy])/2, 1, np.deg2rad(median_rel_dtm[bval_idx,ang_idx,yy]), alpha=0.1, color='g')
            ax.bar(np.deg2rad(-median_acc_sfm[bval_idx,ang_idx,yy])-np.deg2rad(median_rel_sfm[bval_idx,ang_idx,yy])/2, 1, np.deg2rad(median_rel_sfm[bval_idx,ang_idx,yy]), alpha=0.1, color='r')
            # Plot a stick in the middle for each one of these :
            
            ax.plot([np.deg2rad(median_acc_dtm[bval_idx, ang_idx,yy]), #+ np.deg2rad(median_rel_dtm[bval_idx,ang_idx,yy])/2, 
                     np.deg2rad(median_acc_dtm[bval_idx, ang_idx,yy])],# + np.deg2rad(median_rel_dtm[bval_idx,ang_idx,yy])/2, 
                     [0,1], color='g', linewidth=2)

            ax.plot([-np.deg2rad(median_acc_sfm[bval_idx, ang_idx,yy]), #+ np.deg2rad(median_rel_sfm[bval_idx,ang_idx,yy])/2,  
                     -np.deg2rad(median_acc_sfm[bval_idx, ang_idx,yy])], #+ np.deg2rad(median_rel_sfm[bval_idx,ang_idx,yy])/2, 
                     [0,1], color='r', linewidth=2)


fig.set_size_inches([10,6])
fig.savefig('figures/sim_small_mults.svg')

# <codecell>

for yy, w_bal in enumerate(weights):
    fig, ax = plt.subplots(1,len(angles), sharex=True, sharey=True)
    for xx, ang in enumerate(angles):
        ax[xx].plot(bvals_to_test, median_rel_dtm[:,xx,yy],'o-', label='Tensor')
        ax[xx].plot(bvals_to_test, median_rel_sfm[:,xx,yy],'o-', label='SFM')
        #ax[xx].errorbar(bvals_to_test, median_rel_dtm[:,xx,0], color='k', yerr=median_rel_dtm_e[:,xx,0])
        #ax[xx].errorbar(bvals_to_test, median_rel_sfm[:,xx,0], color='k', yerr=median_rel_sfm_e[:,xx,0])
        ax[xx].set_xticks([1000, 2000, 4000])
        ax[xx].set_xlim([0, 5000])
        #ax[xx].text(675, 42, str(int(ang)))
        ax[xx].grid()
        ax[xx].set_xlabel('b value')
    ax[0].set_ylabel('Reliability (degrees)')
    #ax[-1].legend()
    fig.set_size_inches([13, 4])
    fig.savefig('figures/sim_rel.svg')

# <codecell>

for yy, w_bal in enumerate(weights):
    fig, ax = plt.subplots(1,len(angles), sharex=True, sharey=True)
    for xx, ang in enumerate(angles):
        ax[xx].plot(bvals_to_test, median_acc_dtm[:,xx,yy],'o-', label='Tensor')
        ax[xx].plot(bvals_to_test, median_acc_sfm[:,xx,yy],'o-', label='SFM')
        #ax[xx].errorbar(bvals_to_test, median_acc_dtm[:,xx,0], color='k', yerr=median_acc_dtm_e[:,xx,0])
        #ax[xx].errorbar(bvals_to_test, median_acc_sfm[:,xx,0], color='k', yerr=median_acc_sfm_e[:,xx,0])
        ax[xx].set_xticks([1000, 2000, 4000])
        ax[xx].set_xlim([0, 5000])
        #ax[xx].text(675, 42, str(int(ang)))
        ax[xx].grid()
        ax[xx].set_xlabel('b value')
    ax[0].set_ylabel('Accuracy (degrees)')
    #ax[-1].legend()
    fig.set_size_inches([13, 4])
    fig.savefig('figures/sim_acc.svg')

