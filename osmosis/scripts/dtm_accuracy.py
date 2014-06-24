import time
import osmosis.model.dti as dti
import osmosis.predict_n2 as pn
from osmosis.utils import separate_bvals
import nibabel as nib
import os
import numpy as np
import osmosis.utils as ozu

if __name__=="__main__":
    sid = "103414"
    hcp_path = '/biac4/wandell/data/klchan13/hcp_data_q3'
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%sid)

    data_file = nib.load(os.path.join(data_path, "data.nii.gz"))
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_no_vent.nii.gz"))

    data = data_file.get_data()
    wm_data = wm_data_file.get_data()
    wm_idx = np.where(wm_data==1)

    bvals = np.loadtxt(os.path.join(data_path, "bvals"))
    bvecs = np.loadtxt(os.path.join(data_path, "bvecs"))

    bval_list, b_inds, unique_b, bvals_scaled = ozu.separate_bvals(bvals)
    all_b_idx = np.where(bvals_scaled != 0)

    ad_rd = np.loadtxt(os.path.join(data_path, "ad_rd_%s.txt"%sid))
    ad = {1000:ad_rd[0,0], 2000:ad_rd[0,1], 3000:ad_rd[0,2]}
    rd = {1000:ad_rd[1,0], 2000:ad_rd[1,1], 3000:ad_rd[1,2]}

    b_inds_w0 = np.concatenate((b_inds[0], b_inds[3]))
    actual, predicted = pn.kfold_xval_gen(dti.TensorModel, data[..., b_inds_w0],
                                       bvecs[:, b_inds_w0], bvals[b_inds_w0],
                                       10, mask=wm_data)
    cod = ozu.coeff_of_determination(actual, predicted)
    np.save(os.path.join(data_path, "dtm_predict_b3k.npy"), predicted)
    np.save(os.path.join(data_path, "dtm_cod_b3k.npy"), cod)

