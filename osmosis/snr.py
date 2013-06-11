import numpy as np
import osmosis.viz.mpl as mpl
import osmosis.tensor as ozt
import osmosis.model.dti as dti
import nibabel as nib
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import gamma

def separate_bvals(bvals):
    """
    Separates b values into groups with similar values
    
    Returns the grouped b values and their corresponding indices.
    """
    
    if bvals[len(bvals)-1]>4:
        bvals = bvals/1000

    bvals0 = np.array([bvals[bvals < 0.5],np.where(bvals < 0.5)])
    bvals1 = np.array([bvals[np.logical_and(bvals > 0.5, bvals < 1.5)], np.where(np.logical_and(bvals > 0.5,bvals < 1.5))])
    bvals2 = np.array([bvals[np.logical_and(bvals > 1.5, bvals < 2.5)], np.where(np.logical_and(bvals > 1.5, bvals < 2.5))])
    bvals3 = np.array([bvals[bvals > 2.5],np.where(bvals > 2.5)])

    return bvals0, bvals1, bvals2, bvals3

def calculate_snr(data, bvals, b, mask):
    """
    Calculates signal-to-noise ratio (SNR) of the signal at different b values.
    
    Can calculate the SNR on the entire data or just a subset.
    """
    
    if bvals[len(bvals)-1]>4:
        bvals = bvals/1000
        
    b_dict = {1000:1, 2000:2, 3000:3, 1:1, 2:2, 3:3, 0:0}
    if hasattr(mask, 'get_data'):
        mask = mask.get_data()
    if hasattr(data, 'get_data'):
        data = data.get_data()

    # Separate b values
    bvals_sep = separate_bvals(bvals)
    bvals0 = bvals_sep[0]
    this_b = bvals_sep[b_dict[b]]
    
    idx_mask = np.where(mask)
        
    # Initialize the output: 
    out = np.zeros_like(mask)
    
    this_data = data[idx_mask]
    bval_data = this_data[:, this_b[1]]
    b0_data = this_data[:, bvals0[1]]
    
    # Calculate SNR for each voxel
    sigma = np.std(b0_data,-1)
    nb0 = len(bvals_sep[0][0])
    bias = sigma*(1-np.sqrt(2/(nb0-1))*(gamma(nb0/2)/gamma((nb0-1)/2)))
    noise = sigma + bias
    snr_unbiased = np.mean(bval_data, -1)/noise
    
    out[idx_mask] = np.squeeze(snr_unbiased)
    out[np.where(~np.isfinite(out))] = 0
    
    return out

def histogram(data, bvals, bvecs, mask):
    """
    Outputs an probability histogram of SNR
    """
    
    if bvals[len(bvals)-1]>4:
        bvals = bvals/1000
    
    if hasattr(mask, 'get_data'):
        mask = mask.get_data()
        
    # Separate b values
    bvals0, bvals1, bvals2, bvals3 = separate_bvals(bvals)
    idx_mask = np.where(mask)

    prop1 = calculate_snr(data, bvals, 1, mask)[idx_mask]
    prop2 = calculate_snr(data, bvals, 2, mask)[idx_mask]
    prop3 = calculate_snr(data, bvals, 3, mask)[idx_mask]
    all_prop = calculate_all_snr(data, bvals, mask)[idx_mask]
        
    # Median and interquartile range
    prop1_med = np.median(prop1)
    iqr1 = [stats.scoreatpercentile(prop1,25), stats.scoreatpercentile(prop1,75)]
    prop2_med = np.median(prop2)
    iqr2 = [stats.scoreatpercentile(prop2,25), stats.scoreatpercentile(prop2,75)]
    prop3_med = np.median(prop3)
    iqr3 = [stats.scoreatpercentile(prop3,25), stats.scoreatpercentile(prop3,75)]
    prop_all_med = np.median(all_prop)
    iqr_all= [stats.scoreatpercentile(prop3,25), stats.scoreatpercentile(prop3,75)]

    fig = mpl.probability_hist(prop1)
    fig = mpl.probability_hist(prop2, fig = fig)
    fig = mpl.probability_hist(prop3, fig = fig)
    fig = mpl.probability_hist(all_prop, fig = fig)
    plt.legend(('b = 1000', 'b = 2000', 'b = 3000', 'All b values'),loc = 'upper right')
    
def calculate_all_snr(data, bvals, mask):
    """
    Calculates the SNR of each voxel at all the b values.
    """
    
    if hasattr(mask, 'get_data'):
        mask = mask.get_data()
    if hasattr(data, 'get_data'):
        data = data.get_data()

    # Initialize array for displaying SNR
    disp_snr = np.zeros(mask.shape)

    # Get b = 0 indices
    bvals0 = separate_bvals(bvals)[0]
    
    # Find snr across each slice
    new_disp_snr = iter_snr(data, mask, disp_snr, bvals0)
    
    # Save all SNR
    #save_data(new_disp_snr, 'all')
        
    # Display output
    #mean_snr = display_snr(new_disp_snr)

    return new_disp_snr
    
def iter_snr(data, mask, disp_snr, bvals0):
    """
    Iterates through the slices and finds the snr of each voxel
    """
    for m in range(mask.shape[2]):
        # Initialize the mask data, slice data, and isolate data.
        slice_mask = np.zeros(mask.shape)
        slice_mask[:,:,m] = mask[:,:,m]
        slice_data = data[np.where(slice_mask)]
        b0_data = slice_data[:, bvals0[1]]
        
        # Calculate SNRs of the slices and assign into array
        sigma = np.squeeze(np.std(b0_data,-1))
        nb0 = len(bvals0[0])
        bias = sigma*(1-np.sqrt(2/(nb0-1))*(gamma(nb0/2)/gamma((nb0-1)/2)))
        noise = sigma + bias
        snr_unbiased = np.mean(slice_data, -1)/noise      
        disp_snr[np.where(slice_mask)] = snr_unbiased
        
        # Save slice
        #file_slice_data = np.zeros(slice_mask.shape)
        #file_slice_data[np.where(slice_mask)] = bsnr
        #save_data(file_slice_data, "slice", m = m)
        
    disp_snr[~np.isfinite(disp_snr)] = 0
    
    return disp_snr
    
def save_data(data, data_type, m = 'None'):
    """
    Saves the data as a nifti file
    """
    if data_type is "slice":
        ni = nib.Nifti1Image(data,None)
        ni.to_filename("snr_slice{0}.nii.gz".format(m+1))
    elif data_type is "all":
        ni = nib.Nifti1Image(data,None)
        ni.to_filename("all_snr.nii.gz")
        
def display_snr(snr_data):
    """
    Displays the snr across the brain as a mosaic
    """
    #mean_snr = mean(snr_data[np.where(np.isfinite(snr_data))])
    mean_snr = np.mean(snr_data[snr_data>0])
    fig = mpl.mosaic(snr_data, cmap=matplotlib.cm.bone)
    fig.set_size_inches([20,10])

    return mean_snr