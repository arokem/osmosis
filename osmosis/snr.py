import numpy as np
import osmosis.viz.mpl as mpl
import nibabel as nib
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import gamma

def separate_bvals(bvals, mode = None, factor=1000.):
    """
    Separates b values into groups with similar values
    Returns the grouped b values and their corresponding indices.
    
    Parameters
    ----------
    bvals: ndarray
        b values to be separated
    mode: str (optional)
       None: Outputs indices in reference to the original array
       "remove0": Outputs indices in reference to an array not containing b = 0 values.
    factor: float
       This is a tolerance factor. This function will divide the bvalues to the closest 
       factor. That is, when this is 1000, we will get things divided to 1000, 2000, etc.
       
    Returns
    -------
    bval_list: list
        List of separated b values.  Each index contains an array of grouped b values
        with similar values
    bval_ind: list
        List of the indices corresponding to the separated b values.  Each index
        contains an array of the indices to the grouped b values with similar values
    unique_b: 1 dimensional array
        Array of all the unique b values found
    """
    
    bvals = bvals/factor
    
    # Round all the b values and find the unique numbers
    rounded_bvals = list()
    for j in np.arange(len(bvals)):
        if mode is 'remove0':
            if round(bvals[j]) != 0:
                rounded_bvals.append(round(bvals[j]))
        else:
            rounded_bvals.append(round(bvals[j]))
      
    unique_b = np.unique(np.array(rounded_bvals))*factor
    bvals_scaled = np.array(rounded_bvals)*factor
    
    # Initialize one list for b values and another list for the indices
    bval_list = list()
    bval_ind = list()
    
    # Find locations where rounded b values equal the unique b values
    for i in np.arange(len(unique_b)):
      idx_b = np.where(bvals_scaled == unique_b[i])
      bval_list.append(bvals_scaled[idx_b])
      bval_ind.append(idx_b)

    return bval_list, np.squeeze(bval_ind), unique_b, bvals_scaled

def b_snr(data, bvals, b, mask):
    """
    Calculates the signal-to-noise ratio (SNR) of the signal at each voxel at different
    b values.  Can calculate the SNR on the entire data or just a subset of the data if
    given a mask.
    
    Parameters
    ----------
    bval_list: list
        List of separated b values.  Each index contains an array of grouped b values
        with similar values
    bval_ind: list
        List of the indices corresponding to the separated b values.  Each index
        contains an array of the indicies to the grouped b values with similar values
    Returns
    -------
    bsnr: 3 dimensional array
        SNR at each voxel
    """
    
    if hasattr(data, 'get_data'):
        data = data.get_data()
        affine = data.get_affine()
        
    if mask is not 'None':
        if hasattr(mask, 'get_data'):
            mask = mask.get_data()

    # Separate b values
    bval_list, bval_ind, unique_b, bvals_scaled = separate_bvals(bvals)
    bvals0_ind = bval_ind[0]
    this_b_ind = bval_ind[b]
    
    idx_mask = np.where(mask)
        
    # Initialize the output: 
    bsnr = np.zeros_like(mask)
    
    this_data = data[idx_mask]
    bval_data = this_data[:, this_b_ind]
    b0_data = this_data[:, bvals0_ind]
    
    # Calculate SNR for each voxel
    snr_unbiased = calculate_snr(b0_data, bval_data)
    
    bsnr[idx_mask] = np.squeeze(snr_unbiased)
    bsnr[np.where(~np.isfinite(bsnr))] = 0
    
    return bsnr
    
def calculate_snr(b0_data, bval_data):
    """
    Does the SNR calculations and unbiases them.
    
    Parameters
    ----------
    bvals0: ndarray
        b = 0 values
    b0_data: 2 dimensional array
        Data where b = 0
    bval_data: 2 dimensional array
        Data at the desired b value(s)

    Returns
    -------
    snr_unbiased: 1 dimensional array
        Array of the resulting unbiased SNRs
    """
    sigma = np.squeeze(np.std(b0_data,-1))
    nb0 = np.squeeze(b0_data).shape[1]
    bias = sigma*(1-np.sqrt(2/(nb0-1))*(gamma(nb0/2)/gamma((nb0-1)/2)))
    noise = sigma + bias
    snr_unbiased = np.mean(bval_data, -1)/noise
    
    return snr_unbiased

def probability_curve(data, bvals, bvecs, mask):
    """
    Displays a probability histogram of SNR at different b values and at all b values
    as well as the median and interquartile range
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    bvecs: 3 dimensional array
        All the b vectors
    mask: 3 dimensional array
        Brain mask of the data
    """
    
    if hasattr(mask, 'get_data'):
        mask = mask.get_data()
        
    # Separate b values
    bval_list, bval_ind, unique_b, bvals_scaled= separate_bvals(bvals)
    idx_mask = np.where(mask)
    
    unique_b_list = list(unique_b)
    if 0 in unique_b:
      unique_b_list.remove(0)
    
    legend_list = list()
    all_prop = all_snr(data, bvals, mask)[idx_mask]
    all_prop_med = np.median(all_prop)
    iqr_all = [stats.scoreatpercentile(all_prop,25), stats.scoreatpercentile(all_prop,75)]
    fig = mpl.probability_hist(all_prop)
    ax = fig.axes[0]
    ax.text(20.7, 0.10,"All b values: Median = {0}, IQR = {1}".format(round(all_prop_med,2), round(np.abs(iqr_all[0] - iqr_all[1]),2)),horizontalalignment='center',verticalalignment='center')
    legend_list.append('All b values')
    
    idx_array = np.arange(len(unique_b_list))
    txt_height = 0.12
    for l in idx_array:
        prop = b_snr(data, bvals, unique_b_list[l], mask)[idx_mask]
        prop_med = np.median(prop)
        iqr = [stats.scoreatpercentile(prop,25), stats.scoreatpercentile(prop,75)]
        fig = mpl.probability_hist(prop, fig = fig)
        ax = fig.axes[0]
        ax.text(20.7, txt_height,"b = {0}: Median = {1}, IQR = {2}".format(unique_b_list[l]*1000, round(prop_med,2), round(np.abs(iqr[0] - iqr[1]),2)),horizontalalignment='center',verticalalignment='center')
        txt_height = txt_height + 0.02
        legend_list.append('b = {0}'.format(unique_b_list[l]*1000))
    
    plt.legend(legend_list,loc = 'upper right')
    
def all_snr(data, bvals, mask):
    """
    Calculates the SNR of each voxel at all b values.
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    bvals: 1 dimensional array
        All b values
    mask: 3 dimensional array
        Brain mask of the data

    Returns
    -------
    new_disp_snr: 3 dimensional array
        SNR at each voxel
    """
    
    if hasattr(data, 'get_data'):
        data = data.get_data()
        affine = data.get_affine()
        
    if mask is not 'None':
        if hasattr(mask, 'get_data'):
            mask = mask.get_data()

    # Initialize array for displaying SNR
    disp_snr = np.zeros(mask.shape)

    # Get b = 0 indices
    bval_list, bval_ind, unique_b, bvals_scaled = separate_bvals(bvals)
    bvals0_ind = bval_ind[0]
    
    # Find snr across each slice
    new_disp_snr = iter_snr(data, mask, disp_snr, bvals0_ind)

    return new_disp_snr
    
def iter_snr(data, mask, disp_snr, bvals0_ind):
    """
    Iterates through the slices and finds the snr of each voxel
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    mask: 3 dimensional array
        Brain mask of the data
    disp_snr: 3 dimensional array
        Initializes the output to which SNRs will be assigned into
    bvals0: ndarray
        b = 0 values

    Returns
    -------
    disp_snr: 3 dimensional array
	 SNR at each voxel
    """
    for m in range(mask.shape[2]):
        # Initialize the mask data, slice data, and isolate data.
        slice_mask = np.zeros(mask.shape)
        slice_mask[:,:,m] = mask[:,:,m]
        slice_data = data[np.where(slice_mask)]
        b0_data = slice_data[:, bvals0_ind]
        
        # Calculate SNRs of the slices and assign into array        
        snr_unbiased = calculate_snr(b0_data, slice_data)
        disp_snr[np.where(slice_mask)] = snr_unbiased
    
    disp_snr[~np.isfinite(disp_snr)] = 0
    
    return disp_snr
    
def save_data(data, data_type, m = 'None'):
    """
    Saves the data as a nifti file
    
    Parameters
    ----------
    data: 4 dimensional array
        Diffusion MRI data
    data_type: str
        'slice': if saving a slice
        'all': if saving all the data
    m: int
        Integer indicating the slice number if saving slices
    """
    if data_type is "slice":
        ni = nib.Nifti1Image(data,None)
        ni.to_filename("snr_slice{0}.nii.gz".format(m+1))
    elif data_type is "all":
        ni = nib.Nifti1Image(data,None)
        ni.to_filename("all_snr.nii.gz")
        
def display(data):
    """
    Displays the snr across the brain as a mosaic
    
    Saves the data as a nifti file
    
    Parameters
    ----------
    snr_data: 3 dimensional array
        SNR at each voxel
    """
    fig = mpl.mosaic(data, cmap=matplotlib.cm.bone)
    fig.set_size_inches([20,10])

    return mean_snr