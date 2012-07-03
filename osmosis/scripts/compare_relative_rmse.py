"""
Visualize the relative RMSE for different models and comparisons of the
different models in white matter

"""
import os 
import nibabel as ni
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import rc
#For tex rendering in figures: 
#rc('text', usetex=True)
#For not embedding fonts in svg files:
rc('svg', fonttype='none')

import osmosis as oz
import osmosis.viz as viz
reload(viz)
import osmosis.utils as ozu

oz_path = os.path.split(oz.__file__)[0]
data_path =  oz_path + '/data/'
figure_path = '/Users/arokem/projects/osmosis/doc/figures/'

mask_ni = ni.load(data_path + 'FP_white_matter_resamp_to_dwi.nii.gz')
mask_d = mask_ni.get_data()
mask_idx = np.where(mask_d==1)

t1_ni = ni.load(data_path + 'FP_t1_resampled_to_dwi.nii.gz')
t1_d = t1_ni.get_data()

fig_hist_rmse, ax_hist_rmse = plt.subplots(1)
fig_hist_snr, ax_hist_snr = plt.subplots(1)

for bval_idx, bval in enumerate([1000, 2000, 4000]): 

    snr_file_name = '%ssnr_b%s.nii.gz'%(data_path,
                                                 bval)
    rmse_file_name = '%ssignal_rmse_b%s.nii.gz'%(data_path,
                                                 bval)

    signal_rmse = ni.load(rmse_file_name).get_data()
    snr = ni.load(snr_file_name).get_data()
    vol = ozu.nans(signal_rmse.shape)
    vol[mask_idx] = signal_rmse[mask_idx]
    fig = viz.mosaic(t1_d.T[23:-12], cmap=matplotlib.cm.bone, cbar=False)
    fig = viz.mosaic(vol.T[23:-12], fig=fig, cmap=matplotlib.cm.hot)
    fig.set_size_inches([15,10])
    fig.savefig('%ssignal_rmse_b%s.png'%(figure_path,
                                          bval))

    vol = ozu.nans(snr.shape)
    vol[mask_idx] = snr[mask_idx]
    fig = viz.mosaic(t1_d.T[23:-12], cmap=matplotlib.cm.bone, cbar=False)
    fig = viz.mosaic(vol.T[23:-12], fig=fig, cmap=matplotlib.cm.hot)
    fig.set_size_inches([15,10])
    fig.savefig('%ssignal_snr_b%s.png'%(figure_path,
                                          bval))

    rmse_mask = signal_rmse[mask_idx]
    fig_hist_rmse = viz.probability_hist(rmse_mask[np.isfinite(rmse_mask)],
                                         bins=100, fig=fig_hist_rmse,
                                         label='b=%s'%bval)
    
    ax_hist_rmse.set_xlabel('RMSE')
    ax_hist_rmse.set_ylabel('P(RMSE)')
    
    snr_mask = snr[mask_idx]
    fig_hist_snr = viz.probability_hist(snr_mask[np.isfinite(snr_mask)],
                                         bins=100, fig=fig_hist_snr,
                                         label='b=%s'%bval)
    
    ax_hist_snr.set_xlabel('SNR')
    ax_hist_snr.set_ylabel('P(SNR)')

    
ax_hist_rmse.legend()
ax_hist_snr.legend()
fig_hist_rmse.savefig('%ssignal_rmse_hist.png'%figure_path)
fig_hist_snr.savefig('%ssignal_snr_hist.png'%figure_path)

model_names = [
    'TensorModel',
    'CanonicalTensorModel',
    'MultiCanonicalTensorModel',
    'SparseDeconvolutionModel',
    'SphereModel',
    'PointyCanonicalTensorModel',
    'SphericalHarmonicsModel',
    'PointyMultiCanonicalTensorModel',
    'SparseKernelModel'
            ]

for model_name in model_names:

    fig_hist, ax_hist = plt.subplots(1) 
    for bval_idx, bval in enumerate([1000, 2000, 4000]): 
        print "bvalue = %s, model= %s"%(bval, model_name)
        rmse_file_name = '%s%s_relative_rmse_b%s.nii.gz'%(data_path,
                                                          model_name,
                                                          bval)
        
        rmse = ni.load(rmse_file_name).get_data()
        vol = ozu.nans(rmse.shape)
        vol[mask_idx] = rmse[mask_idx]
        # Clip slices for focus on the main part of things:
        fig = viz.mosaic(t1_d.T[29:60][::2], cmap=matplotlib.cm.bone, cbar=False)
        fig = viz.mosaic(vol.T[29:60][::2], fig=fig, cmap=matplotlib.cm.RdYlGn_r,
                         vmin=0.5, vmax=1.5)

        fig.axes[0].set_axis_bgcolor('black')

        fig.set_size_inches([15,10])
        
        fig.savefig('%s%s_relative_rmse_b%s.png'%(figure_path,
                                                  model_name,
                                                  bval))

        
        rmse_mask = rmse[mask_idx]
        # scale the histograms:
        fig_hist = viz.probability_hist(rmse_mask[np.isfinite(rmse_mask)],
                                        fig=fig_hist, bins=100,
                                        label='b=%s'%bval)
        
        ax_hist.set_xlim([0,4])
        ax_hist.set_xlabel(r'$\frac{RMSE_{model \rightarrow signal}}{RMSE_{signal \rightarrow signal}}$')
        ax_hist.set_ylabel(r'$P(\frac{RMSE_{model \rightarrow signal}}{RMSE_{signal \rightarrow signal}}$)')

        print "%s voxels above 1"%len(np.where(rmse_mask>1)[0])
        
    ax_hist.legend()
    fig_hist.savefig('%s%s_relative_rmse_hist.png'%(figure_path,
                                                      model_name))

        

for bval_idx, bval in enumerate([1000, 2000, 4000]):
    for model1_idx in range(len(model_names)):
        for model2_idx in range(model1_idx+1, len(model_names)):
            model1 = model_names[model1_idx]
            model2 = model_names[model2_idx]
            model1_rmse = ni.load(
            "%s%s_relative_rmse_b%s.nii.gz"%(data_path, model1, bval)).get_data()
            model2_rmse = ni.load(
            "%s%s_relative_rmse_b%s.nii.gz"%(data_path, model2, bval)).get_data()
    
            diff = ozu.nans(mask_d.shape)
            diff[mask_idx] = (model2_rmse - model1_rmse)[mask_idx]

            fig = viz.mosaic(t1_d.T[29:60][::2],
                             cmap=matplotlib.cm.bone, cbar=False)
            vmax = np.nanmax([np.abs(np.nanmin(diff)),
                              np.nanmax(diff)])

            fig = viz.mosaic(diff.T[29:60][::2], fig=fig,
                             vmax=vmax/2, vmin=-1*vmax/2,
                            cmap=matplotlib.cm.RdBu_r)

            fig.set_size_inches([25,20])

            fig.savefig('%sdiff_%s_%s_rrmse_b%s.png'%(figure_path,
                                                           model1,
                                                           model2,
                                                           bval))

            mask_rmse_1 = model1_rmse[mask_idx]
            fig = viz.probability_hist(mask_rmse_1[np.isfinite(mask_rmse_1)],
                                       label=model_names[model1_idx])

            mask_rmse_2 = model2_rmse[mask_idx]
            fig = viz.probability_hist(mask_rmse_2[np.isfinite(mask_rmse_2)],
                                       fig=fig,
                                       label=model_names[model2_idx])

            ax = fig.get_axes()[0]
            ax.set_xlabel(r'$\frac{RMSE_{model \rightarrow signal}}{RMSE_{signal \rightarrow signal}}$')
            ax.set_ylabel(r'$P(\frac{RMSE_{model \rightarrow signal}}{RMSE_{signal \rightarrow signal}}$)')
            
            plt.legend()
            fig.savefig('%sdiff_%s_%s_rrmse_hist_b%s.png'%(figure_path,
                                                           model1,
                                                           model2,
                                                           bval))

            
