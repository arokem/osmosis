import osmosis.model.sparse_deconvolution as sfm
import nibabel as nib
import numpy as np
import osmosis.utils as ozu
import osmosis.snr as snr
import time
import os

def place_predict(files):
    data_path = "/biac4/wandell/data/klchan13/100307/Diffusion/data"

    # Get file object
    data_file = nib.load(os.path.join(data_path, "data.nii.gz"))
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_registered.nii.gz"))

    # Get data and indices
    data = data_file.get_data()
    wm_data = wm_data_file.get_data()
    wm_idx = np.where(wm_data==1)

    # b values
    bvals = np.loadtxt(os.path.join(data_path, "bvals"))
    bval_list, b_inds, unique_b, rounded_bvals = snr.separate_bvals(bvals/1000)
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))

    all_predict_brain = ozu.nans((wm_data_file.shape + bvals.shape))
    bvals_predict_brain = ozu.nans((wm_data_file.shape + bvals.shape))
    
    # Keep track of files in case there are any missing files
    i_track = np.ones(1830)
    for f_idx in np.arange(len(files)):
        this_file = files[f_idx]
        
        predict_data = nib.load(this_file).get_data()
        if this_file[0:11] == "all_predict":
            i = int(this_file.split(".")[0][11:])
            print "Placing all_predict %4.2f of 1830"%(i+1)
            low = i*70
            high = np.min([(i+1) * 70, int(np.sum(wm_data))])
            all_predict_brain[wm_idx[0][low:high], wm_idx[1][low:high], wm_idx[2][low:high]] = predict_data
        elif this_file[0:13] == "bvals_predict":
            i = int(this_file.split(".")[0][13:])
            print "Placing bvals_predict %4.2f of 1830"%(i+1)
            low = i*70
            high = np.min([(i+1) * 70, int(np.sum(wm_data))])
            bvals_predict_brain[wm_idx[0][low:high], wm_idx[1][low:high], wm_idx[2][low:high]] = predict_data
        i_track[i] = 0
        
    actual = data[wm_idx, :][:, all_b_idx]
    missing_files = np.where(i_track)
    rmse_b = np.sqrt(np.mean((actual - all_predict_brain[wm_idx])**2,-1))
    rmse_mb = p.sqrt(np.mean((actual - bvals_predict_brain[wm_idx])**2,-1))

    # Save the rmse and predict data
    aff = data_file.get_affine()
    nib.Nifti1Image(all_predict_brain, aff).to_filename("all_predict_brain.nii.gz")
    nib.Nifti1Image(bvals_predict_brain, aff).to_filename("bvals_predict_brain.nii.gz")

    rmse_aff = np.eye(4)
    nib.Nifti1Image(rmse_b_flat, rmse_aff).to_filename("rmse_b_flat.nii.gz")
    nib.Nifti1Image(rmse_mb_flat, rmse_aff).to_filename("rmse_mb_flat.nii.gz")
    
    return missing_files, all_predict_brain, bvals_predict_brain, rmse_b_flat, rmse_mb_flat