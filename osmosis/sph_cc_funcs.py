"""

Some functions that call to sph_cc in osmosis utilities.

"""

import osmosis.utils as ozu
import numpy as np
import itertools

def sph_cc_ineq(cod_all_mod, cod_bval_mod, bvals, all_thresh, bval_thresh, tol = 0.1):
    """
    Helper function to find the indices where inequalities occur between two given
    input arrays and to separate the b values.
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_inds = np.where(rounded_bvals != 0)
    
    if all_thresh > bval_thresh:
        inds = np.where((cod_all_mod > all_thresh) &
                        (cod_bval_mod < bval_thresh) &
                        (cod_bval_mod > bval_thresh - tol))
    elif all_thresh < bval_thresh:
        inds = np.where((cod_all_mod < all_thresh) &
                        (cod_all_mod > all_thresh - tol) &
                        (cod_bval_mod > bval_thresh))
    elif all_thresh == bval_thresh:
        inds = np.where((cod_all_mod < all_thresh + tol/2) &
                        (cod_all_mod > all_thresh - tol/2) &
                        (cod_bval_mod < bval_thresh + tol/2) &
                        (cod_bval_mod > bval_thresh - tol/2))
        
    return inds, b_inds, all_b_inds
    
def across_sph_cc(cod_all_mod, cod_bval_mod, vol_b_list, bvals,
                bvecs, mask, all_thresh, bval_thresh, idx = None,
                vol_mp_all = None, tol = 0.1, n = 20, ri = None):
    """
    Calculates the spherical cross correlation at a certain index for all b values fit
    together and b values fit separately.
    """
    
    inds, b_inds, all_b_inds = sph_cc_ineq(cod_all_mod, cod_bval_mod, bvals,
                                           all_thresh, bval_thresh, tol)
    
    if idx is not None:
        if ri is None:
            ri = np.random.randint(0, len(inds[0]))
        else:
            ri = ri
        idx = inds[0][ri]
    
    data_list = []
    bvecs_b_list = []
    deg_list = []
    cc_list = []
    
    pool = np.arange(len(vol_b_list))
    
    for ii in pool:
        data_list.append(np.concatenate((vol_b_list[ii][np.where(mask)][idx],
                                         vol_b_list[ii][np.where(mask)][idx]), -1))
        bvecs_b_list.append(np.squeeze(np.concatenate((bvecs[:, b_inds[ii+1]],
                                           -1*bvecs[:, b_inds[ii+1]]), -1)).T)
    
    if vol_mp_all is None:
        combos = list(itertools.combinations(pool, 2))
        this_iter = np.arange(len(combos))
    else:
        combos = None
        this_iter = np.arange(len(vol_b_list))
        bvecs_all = np.squeeze(np.concatenate((bvecs[:, all_b_inds],
                               -1*bvecs[:, all_b_inds]), -1)).T
        data_all = np.concatenate((vol_mp_all[np.where(mask)][idx],
                                   vol_mp_all[np.where(mask)][idx]), -1)
    
    for itr in this_iter:
        if vol_mp_all is None:
            inputs = [np.squeeze(data_list[combos[itr][0]]), np.squeeze(data_list[combos[itr][1]]),
                      bvecs_b_list[combos[itr][0]], bvecs_b_list[combos[itr][1]]]
        else:
            inputs = [np.squeeze(data_all), np.squeeze(data_list[itr]), bvecs_all, bvecs_b_list[itr]]
        
        deg, cc = ozu.sph_cc(*inputs, n = n)
        deg_list.append(deg)
        cc_list.append(cc)
    
    return deg_list, cc_list, combos, idx, cod_all_mod[idx], cod_bval_mod[idx]
    
def all_across_sph_cc(cod_all_mod, cod_bval_mod, vol_b_list, bvals,
                      bvecs, mask, all_thresh, bval_thresh, idx = None,
                      vol_mp_all = None, tol = 0.1, n = 20, ri = None):
                          
    """
    Calculates the spherical cross correlation at different indices for all b values
    fit to gether and b values fit separately.
    """
    
    inds, b_inds, all_b_inds = sph_cc_ineq(cod_all_modulation, cod_bval_modulation, bvals,
                                           all_thresh, bval_thresh, tol = 0.1)

    all_deg_list = []
    all_cc_list = []
    ai_count = 0
    for vol_b in np.arange(len(vol_b_list)):
        all_deg_list.append(np.zeros((len(inds[0]), n-1)))
        all_cc_list.append(np.zeros((len(inds[0]), n-1)))
        
    for ai in inds[0]:
        [deg_list, cc_list, combos, idx,
         cod_all_idx, cod_bval_idx] = across_sph_cc(cod_all_modulation, cod_bval_modulation,
                                                    vol_b_list, bvals, bvecs, wm_data,
                                                    all_thresh, bval_thresh, idx = ai)
        for dli in np.arange(len(deg_list)):
            all_deg_list[dli][ai_count] = deg_list[dli]
            all_cc_list[dli][ai_count] = cc_list[dli]
            
        ai_count = ai_count + 1
        
    return all_deg_list, all_cc_list