
# Lines above this one are auto-generated by the wrapper to provide as params:
# i, sid, fODF, im, data_path
import time
import osmosis.multi_bvals as sfm_mb
import osmosis.model.dti as dti
import osmosis.predict_n as pn
from osmosis.utils import separate_bvals
import osmosis.mean_diffusivity_models as mdm
import nibabel as nib
import os
import numpy as np

if __name__=="__main__":
    t1 = time.time()
    
    data_file = nib.load(os.path.join(data_path, "data.nii.gz"))
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_no_vent.nii.gz"))
    
    data = data_file.get_data()
    mask_data = wm_data_file.get_data()
    mask_idx = np.where(mask_data==1)
    
    bvals = np.loadtxt(os.path.join(data_path, "bvals"))
    bvecs = np.loadtxt(os.path.join(data_path, "bvecs"))
    
    low = i*2000
    # Make sure not to go over the edge of the mask:
    high = np.min([(i+1) * 2000, int(np.sum(mask_data))])

    # Now set the mask:
    mask = np.zeros(wm_data_file.shape)
    mask[mask_idx[0][low:high], mask_idx[1][low:high], mask_idx[2][low:high]] = 1
    
    if im == "bi_exp_rs":
        shorthand_im = "be"
    elif im == "single_exp_rs":
        shorthand_im = "se"
        
    param_out, fit_out, _ = mdm.optimize_MD_params(data, bvals, bvecs, mask,
                                                   im, signal = "relative_signal")
    
    cod, predict_out = mdm.kfold_xval_MD_mod(data, bvals, bvecs, mask,
                                            im, 10, signal="relative_signal")
    
    
    np.save(os.path.join(data_path, "im_cod_%s%s.npy"%(shorthand_im, i)), cod)
    np.save(os.path.join(data_path, "im_predict_out_%s%s.npy"%(shorthand_im, i)), predict_out)
    np.save(os.path.join(data_path, "im_param_out_%s%s.npy"%(shorthand_im, i)), param_out)
    
    t2 = time.time()
    print "This program took %4.2f minutes to run."%((t2 - t1)/60.)
