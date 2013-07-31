import osmosis.model.sparse_deconvolution as sfm
import osmosis.snr as snr
import osmosis.multi_bvals as sfm_mb
import numpy as np
import time

# Predict for all b values
def predict_n(data, bvals, bvecs, mask, n, b_mode):
    """
    Predict like before but for 10% instead of each one.
    
    Parameters
    ----------
    
    data: 
    
    ...
       
    """ 
    t1 = time.time()
    bval_list, b_inds, unique_b, rounded_bvals = snr.separate_bvals(bvals)
    _, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = snr.separate_bvals(bvals,
                                                                        mode = 'remove0')
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    
    actual = np.empty((np.sum(mask), len(all_b_idx)))
    predicted = np.empty(actual.shape)
    
    if b_mode == 'all': 
        full_mod = sfm_mb.SparseDeconvolutionModelMultiB(data, bvecs, bvals,
                                                        mask = mask,
                                                        params_file = "temp")                 
    for bi in np.arange(len(unique_b[1:])):
        
        if b_mode is "all":
            all_inc_0 = np.arange(len(rounded_bvals))
            bvals_pool = bvals
        elif b_mode is "bvals":
            all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][bi]))
            bvals_pool = rounded_bvals
        
        these_b_inds = b_inds[1:][bi]
        these_b_inds_rm0 = b_inds_rm0[bi]
        vec_pool = np.arange(len(these_b_inds))
        
        # Need to choose random indices so shuffle them!
        np.random.shuffle(vec_pool)
        
        # How many of the indices are you going to leave out at a time?
        num_choose = (n/100.)*len(these_b_inds)
                
        for combo_num in np.arange(np.floor(100./n)):
            these_inc0 = list(all_inc_0)
            idx = list(b_inds_rm0[bi])
            vec_pool_inds = vec_pool[(combo_num)*num_choose:(combo_num*num_choose + num_choose)]
            vec_combo = these_b_inds[vec_pool_inds]
            vec_combo_rm0 = these_b_inds_rm0[vec_pool_inds]
               
            # Remove the chosen indices from the rest of the indices
            for choice_idx in vec_pool_inds:
                these_inc0.remove(these_b_inds[choice_idx])
                idx.remove(these_b_inds_rm0[choice_idx])
            
            for b_idx in np.arange(len(unique_b[1:])):
                if np.logical_and(b_idx != bi, b_mode is "all"):
                    idx = np.concatenate((idx, b_inds_rm0[b_idx]),0)
                
            # Make the list back into an array
            these_inc0 = np.array(these_inc0)
            
            # Isolate the b vectors, b values, and data not including those to be predicted
            these_bvecs = bvecs[:, these_inc0]
            these_bvals = bvals_pool[these_inc0]
            this_data = data[:, :, :, these_inc0]
            
            # Need to sort the indices first before indexing full_mod.regressors
            si = sorted(idx)
            
            if b_mode is "all":
                mod = sfm_mb.SparseDeconvolutionModelMultiB(this_data, these_bvecs, these_bvals,
                                                            mask = mask, params_file = "temp")
                                                            
                mod.regressors = [full_mod.regressors[0][:, si],
                                  full_mod.regressors[1][:, si][si, :],
                                  full_mod.regressors[2][:, si][si, :],
                                  full_mod.regressors[3][:, si],
                                  full_mod.regressors[4][:, si]]
                                  
                predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo], bvals[vec_combo])
                
            elif b_mode is "bvals":
                mod = sfm.SparseDeconvolutionModel(this_data, these_bvecs, these_bvals,
                                                    mask = mask, params_file = "temp")
                                       
                predicted[:, vec_combo_rm0] = mod.predict(bvecs[:, vec_combo])[mod.mask]
                
            actual[:, vec_combo_rm0] = data[mod.mask][:, vec_combo]
            
        t2 = time.time()
        print "This program took %4.2f minutes to run through %3.2f percent of the program."%((t2 - t1)/60,
                                                                100*((bi+1)/len(unique_b[1:])))
    return actual, predicted
    
def predict_bvals(data, bvals, bvecs, mask, b_fit_to, b_predict):
    """
    Predict for each b value.
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = snr.separate_bvals(bvals)
    bval_list_rm0, b_inds_rm0, unique_b_rm0, rounded_bvals_rm0 = snr.separate_bvals(bvals,
                                                                        mode = 'remove0')
    all_b_idx = np.squeeze(np.where(rounded_bvals != 0))
    all_inc_0 = np.concatenate((b_inds[0], b_inds[1:][b_fit_to]))
        
    mod = sfm.SparseDeconvolutionModel(data[:,:,:,all_inc_0], bvecs[:,all_inc_0],
                                               rounded_bvals[all_inc_0], mask = mask,
                                               params_file = 'temp')
    actual = data[mod.mask][0, b_inds[b_predict]]
    this_pred = mod.predict(bvecs[:, b_inds[b_predict]])[mod.mask][0]
    predicted = this_pred
        
    return actual, predicted