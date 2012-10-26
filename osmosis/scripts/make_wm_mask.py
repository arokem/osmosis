#!/usr/bin/python
"""
make_wm_mask

Script to make a white-matter mask from an itk-gray class file and DWI data, at
the resolution of the DWI data. 

This follows [Jeurissen2012]_. First, we resample a segmentation based on a
T1-weighted image. Then, we calculate the mean diffusivity in a coregistered
DWI image. We then also exclude from the segmentation white matter voxels that
have

     MD > median (MD) + 2 * IQR (MD)

Where IQR is the interquartile range. Note that we take a slightly more
restrictive criterion (of 2 * IQR, instead of 1.5 * IQR), based on some
empirical looking at data. Not so sure why there would be a difference.

[Jeurissen2012]Jeurissen B, Leemans A, Tournier, J-D, Jones, DK and Sijbers J
(2012). Investigating the prevalence of complex fiber configurations in white
matter tissue with diffusion magnetic resonance imaging. Human Brain Mapping
doi: 10.1002/hbm.2209

"""

import sys

import numpy as np
import scipy.stats as stats

import nibabel as ni
from nipy.labs.datasets import as_volume_img

import osmosis.model as ozm

if "__name__"=="__main__":
    seg_path = sys.argv[0]
    dwi_path = sys.argv[1] # This should lead to nii.gz, bvecs and bvals
    out_path = sys.argv[2]
    
    # Load the classification information: 
    seg_ni = ni.load(seg_path)
    seg_vimg = as_volume_img(seg_ni)
    dwi_ni = ni.load(dwi_path + '.nii.gz')

    vimg = as_volume_img(seg_ni)
    # This does the magic - resamples using nearest neighbor interpolation:
    seg_resamp_vimg = vimg.as_volume_img(affine=dwi_ni.get_affine(),
                                         shape=dwi_ni.get_shape()[:-1],
                                         interpolation='nearest')

    seg_resamp = seg_resamp_vimg.get_data()

    # We know that WM is classified as 3 and 4 (for the two different
    # hemispheres):
    wm_idx = np.where(np.logical_or(seg_resamp==3, seg_resamp==4))
    vol = np.zeros(seg_resamp.shape)
    vol[wm_idx] = 1

    # OK - now we need to find and exclude MD outliers: 
    TM = ozm.TensorModel(dwi_path + '.nii.gz',
                         dwi_path + '.bvecs',
                         dwi_path + '.bvals', mask=vol,
                         params_file='temp')    

    MD = TM.mean_diffusivity
    
    IQR = (stats.scoreatpercentile(MD[np.isfinite(MD)],75) -
          stats.scoreatpercentile(MD[np.isfinite(MD)],25))
    cutoff = np.median(MD[np.isfinite(MD)]) + 2 * IQR
    cutoff_idx = np.where(MD > cutoff)

    # Null 'em out:
    vol[cutoff_idx] = 0

    # Now, let's save some output:
    ni.Nifti1Image(vol, dwi_ni.get_affine).to_filename(out_path)
    
    
