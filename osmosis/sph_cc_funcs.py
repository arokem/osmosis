"""

Some functions that call to sph_cc in osmosis utilities.

"""

import osmosis.utils as ozu
import numpy as np
import os
import itertools
import nibabel as nib

def sph_cc_ineq(cod_single_mod, cod_multi_mod, bvals, single_thresh, multi_thresh, tol = 0.1):
    """
    Helper function to find the indices where inequalities occur between two given
    input arrays and to separate the b values.
    
    Parameters
    ----------
    cod_single_mod: 1 dimensional array
        Coefficient of determination at each voxel for the single fODF model
    cod_multi_mod: 1 dimensional array
        Coefficient of determination at each voxel for the multi fODF model
    bvals: 1 dimensional array
        All b values
    single_thresh: int
        Coefficient of determination threshold for the single fODF model
    multi_thresh: int
        Coefficient of determination threshold for the multi fODF model
    tol: float
        Tolerance
        
    Returns
    -------
    inds: 1 dimensional array
        Indices indicating the voxels within the COD inequality/equality
    b_inds: list
        List of indices corresponding to each b value
    all_b_inds: 1 dimensional array
        Indices corresponding to the non-zero b values
    """
    
    bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
    all_b_inds = np.where(rounded_bvals != 0)
    
    if single_thresh > multi_thresh:
        inds = np.where((cod_single_mod > single_thresh) &
                        (cod_multi_mod < multi_thresh) &
                        (cod_multi_mod > multi_thresh - tol))
    elif single_thresh < multi_thresh:
        inds = np.where((cod_single_mod < single_thresh) &
                        (cod_single_mod > single_thresh - tol) &
                        (cod_multi_mod > multi_thresh))
    elif single_thresh == multi_thresh:
        inds = np.where((cod_single_mod < single_thresh + tol/2) &
                        (cod_single_mod > single_thresh - tol/2) &
                        (cod_multi_mod < multi_thresh + tol/2) &
                        (cod_multi_mod > multi_thresh - tol/2))
        
    return np.squeeze(inds), b_inds, all_b_inds
    
def across_sph_cc(vol_b_list, bvals, bvecs, mask, cod_single_mod = None, cod_multi_mod = None,
                  single_thresh = None, multi_thresh = None, idx = None, vol_mp_single = None,
                  tol = 0.1, n = 20):
    """
    Calculates the spherical cross correlation at a certain index for all b values fit
    together and b values fit separately.
    
    Parameters
    ----------
    vol_b_list: list
        List of the model parameters for each voxel for an fODF fit to each b value.
    bvals: 1 dimensional array
        All b values
    bvecs: 2 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    cod_single_mod: 1 dimensional array
        Coefficient of determination at each voxel for the single fODF model
    cod_multi_mod: 1 dimensional array
        Coefficient of determination at each voxel for the multi fODF model
    single_thresh: int
        Coefficient of determination threshold for the single fODF model
    multi_thresh: int
        Coefficient of determination threshold for the multi fODF model
    idx: int
        Index into the indices indicating the voxels included in the COD (in)equality
    vol_mp_single: 2 dimensional array
        Model parameters from fitting a single fODF to each voxel
    tol: float
        Tolerance for the COD (in)equality
    n: int
        Integer indicating the number of directions to divide by for spherical
        cross-correlation
    
    Returns
    -------
    deg_list: list
        List indicating the degrees included in spherical cross-correlation
    cc_list: list
        List with the cross-correlations between each combination of b values
    idx: int
        Index into the indices indicating the voxels included in the COD (in)equality
    cod_s: float
        Coefficient of determination of single fODF model
    cod_m: float
        Coefficient of determination of multi fODF model
    """
    if (single_thresh != None) & (multi_thresh != None):
        # Get the indices with a desired COD.
        inds, b_inds, all_b_inds = sph_cc_ineq(cod_single_mod, cod_multi_mod, bvals,
                                            single_thresh, multi_thresh, tol = tol)
    else:
        # With no COD threshold, just find all the indices.
        inds = np.arange(int(np.sum(mask)))
        bval_list, b_inds, unique_b, rounded_bvals = ozu.separate_bvals(bvals)
        all_b_inds = np.where(rounded_bvals != 0)
    
    if idx is None:
        # If a specific index is not given, just find a random index.
        if ri is None:
            ri = np.random.randint(0, len(inds))
        else:
            ri = ri
        idx = inds[ri]
    
    data_list = []
    bvecs_b_list = []
    deg_list = []
    cc_list = []
    
    pool = np.arange(len(vol_b_list))
    
    for ii in pool:
        # Just get the data (model parameters) and bvecs within the chosen voxel (idx)
        # for each b value and mirror them.
        data_list.append(np.concatenate((vol_b_list[ii][np.where(mask)][idx],
                                         vol_b_list[ii][np.where(mask)][idx]), -1))
        bvecs_b_list.append(np.squeeze(np.concatenate((bvecs[:, b_inds[ii+1]],
                                           -1*bvecs[:, b_inds[ii+1]]), -1)).T)
    if vol_mp_single is None:
        # Make combinations of b values for spherical cross correlation between b values
        combos = list(itertools.combinations(pool, 2))
        this_iter = np.arange(len(combos))
    else:
        # No need for combos since comparing between the the single fODF and the multi fODF
        combos = None
        this_iter = np.arange(len(vol_b_list))
        bvecs_all = np.squeeze(np.concatenate((bvecs[:, all_b_inds],
                               -1*bvecs[:, all_b_inds]), -1)).T
        data_all = np.concatenate((vol_mp_single[np.where(mask)][idx],
                                   vol_mp_single[np.where(mask)][idx]), -1)
    
    for itr in this_iter:
        if vol_mp_single is None:
            # Inputs are data and bvecs from two different b values that you want to find the
            # spherical cross-correlation between
            inputs = [np.squeeze(data_list[combos[itr][0]]), np.squeeze(data_list[combos[itr][1]]),
                      bvecs_b_list[combos[itr][0]], bvecs_b_list[combos[itr][1]]]
        else:
            # Inputs are the data and bvecs from one b value and the single fODF
            inputs = [np.squeeze(data_all), np.squeeze(data_list[itr]), bvecs_all, bvecs_b_list[itr]]
        # Put the inputs into the spherical cross-correlation function
        deg, cc = ozu.sph_cc(*inputs, n = n)
        deg_list.append(deg)
        cc_list.append(cc)
    
    if (single_thresh != None) & (multi_thresh != None):
        # Because sometimes it's nice to know what the actual CODs are
        cod_s = cod_single_mod[idx]
        cod_m = cod_multi_mod[idx]
    else:
        cod_s = None
        cod_m = None

    return deg_list, cc_list, combos, idx, cod_s, cod_m
    
def all_across_sph_cc(vol_b_list, bvals, bvecs, mask, cod_single_mod = None,
                      cod_multi_mod = None, single_thresh = None,
                      multi_thresh = None, vol_mp_single = None,
                      tol = 0.1, n = 20):  
    """
    Calculates the spherical cross correlation at different indices for all b values
    fit to gether and b values fit separately.
    
    Parameters
    ----------
    vol_b_list: list
        List of the model parameters for each voxel for an fODF fit to each b value.
    bvals: 1 dimensional array
        All b values
    bvecs: 2 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    cod_single_mod: 1 dimensional array
        Coefficient of determination at each voxel for the single fODF model
    cod_multi_mod: 1 dimensional array
        Coefficient of determination at each voxel for the multi fODF model
    single_thresh: int
        Coefficient of determination threshold for the single fODF model
    multi_thresh: int
        Coefficient of determination threshold for the multi fODF model
    vol_mp_single: 2 dimensional array
        Model parameters from fitting a single fODF to each voxel
    tol: float
        Tolerance for the COD (in)equality
    n: int
        Integer indicating the number of directions to divide by for spherical
        cross-correlation
    
    Returns
    -------
    all_deg_list: list
        List indicating the degrees included in spherical cross-correlation for
        each voxel
    all_cc_list: list
        List with the cross-correlations at each voxel for each combination of
        b values
    """
    
    if (single_thresh != None) & (multi_thresh != None):
        # Get the indices with a desired COD.
        inds, b_inds, all_b_inds = sph_cc_ineq(cod_single_mod, cod_multi_mod, bvals,
                                            single_thresh, multi_thresh, tol = 0.1)
    else:
        # Just get all the indices.
        inds = np.arange(int(np.sum(mask)))

    all_deg_list = []
    all_cc_list = []
    for vol_b in np.arange(len(vol_b_list)):
        # Preallocate the arrays in each list.
        all_deg_list.append(np.zeros((len(inds), n-1)))
        all_cc_list.append(np.zeros((len(inds), n-1)))
        
    for ai_count, ai in enumerate(inds):
        # Call to the main spherical cc function between the fODFs of different b values
        # or between the single fODF and the fODFs of other b values for each voxel.
        [deg_list, cc_list, combos,
        idx, cod_s, cod_m]= across_sph_cc(vol_b_list, bvals, bvecs, mask,
                                          cod_single_mod = cod_single_mod,
                                          cod_multi_mod = cod_multi_mod,
                                          single_thresh = single_thresh,
                                          multi_thresh = multi_thresh,
                                          idx = ai)
        for dli in np.arange(len(deg_list)):
            all_deg_list[dli][ai_count] = deg_list[dli]
            all_cc_list[dli][ai_count] = cc_list[dli]
        
    return all_deg_list, all_cc_list
    
def place_files(file_names, mask_vox_num, expected_file_num, mask_data_file,
                 file_path = os.getcwd(), vol = "No",
                 f_type = "npy",save = "No", num_dirs = 270):
    """
    Function to aggregate sub data files from parallelizing.  Assumes that
    the sub_files are in the format: (file name)_(number of sub_file).(file_type)
    
    Parameters
    ----------
    file_names: list
        List of strings indicating the base file names for each output data
        aggregation
    mask_vox_num: int
        Number of voxels in each sub file
    expected_file_num: int
        Expected number of sub files
    mask_data_file: obj
        File handle for brain/white matter mask
    file_path: str
        Path to the directory with all the sub files.  Default is the current
        directory
    vol: str
        String indicating whether or not the sub files are in volumes and
        whether the output files are saved as volumes as well
    f_type: str
        String indicating the type of file the sub files are saved as
    save: str
        String indicating whether or not to save the output aggregation/volumes
    num_dirs: int
        Number of directions in each output aggregation/volume
    
    Results
    -------
    missing_files: 1 dimensional array
        All the sub files that are missing in the aggregation
    aggre_list: list
        List with all the aggregations/volumes
    """
    files = os.listdir(file_path)

    # Get data and indices
    mask_data = mask_data_file.get_data()
    mask_idx = np.where(mask_data)
    
    aggre_list = []
    # Keep track of files in case there are any missing ones
    i_track = np.ones(expected_file_num)
    for fn in file_names:
        # If you don't want to put the voxels back into a volume, just preallocate
        # enough for each voxel included in the mask.
        if vol == "No":
            aggre = np.squeeze(np.zeros((int(np.sum(mask_data)),) + (num_dirs,)))
        else:
            aggre = np.squeeze(ozu.nans((mask_data_file.shape + (num_dirs,))))
            
        for f_idx in np.arange(len(files)):
            this_file = files[f_idx]
            if this_file[(len(this_file)-len(f_type)):len(this_file)] == f_type:
                
                if f_type == "npy":
                    sub_data = np.load(os.path.join(file_path, this_file))
                elif f_type == "nii.gz":
                    sub_data = nib.load(os.path.join(file_path, this_file)).get_data()
                # If the name of this file is equal to file name that you want to
                # aggregate, load it and find the voxels corresponding to its location
                # in the given mask.
                if this_file[0:len(fn)] == fn:
                    i = int(this_file.split(".")[0][len(fn):])
                    low = i*mask_vox_num
                    high = np.min([(i+1) * mask_vox_num, int(np.sum(mask_data))])
                    
                    if vol == "No":
                        aggre[low:high] = sub_data.T
                    else:
                        mask = np.zeros(mask_data_file.shape)
                        mask[mask_idx[0][low:high], mask_idx[1][low:high], mask_idx[2][low:high]] = 1
                        aggre[np.where(mask)] = sub_data
                    # If the file is present, change its index within the tracking array to 0.    
                    i_track[i] = 0

        aggre_list.append(aggre)
        
        if save == "Yes":
            if vol == "No":
                np.save("aggre_%s.npy"%fn, aggre)
            else:
                aff = mask_data_file.get_affine()
                nib.Nifti1Image(aggre, aff).to_filename("vol_%s.nii.gz"%fn)            
            
    missing_files = np.squeeze(np.where(i_track))
    
    return missing_files, aggre_list
