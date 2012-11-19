"""
osmosis.volume
=================

Integration of data from volumes and into volumes


"""
import os

import numpy as np
import nibabel as ni
from nipy.labs.datasets import as_volume_img

import osmosis.fibers as ozf
import osmosis.utils as ozu


def nii2fg(fg, nii, data_node=0, stat_name=None):
    """
    Attach data from a nifti volume to the fiber-group fiber_stats dict

    Parameters
    ----------
    fg: A osmosis.fibers.FiberGroup class instance or a full path to a pdb
        file.

    nii: A nibabel.Nifti1 class instance or a full path to a nifti file.

    data_node: Whether to use data from the beginning (0) or end (-1) of each
    fiber as the source of the fiber statistic. Defaults to 0

    stat_name: What will be the name of the statistic in the fiber group (this
    will be the key into the FiberGroup fiber_stats dict.

    Returns
    -------
    fg: FiberGroup class instance with the generated statistic as one of the
    fiber_stat dictionary values.
    
    """

    if data_node not in [0,-1]:
        e_s = "Can only attach data from voxels near the"
        e_s += " beginning or end of the fiber"
        raise ValueError(e_s)
    
    if stat_name is None:
        if isinstance(nii, str):
            stat_name = os.path.split(nii)[-1]
        else:
            stat_name = 'stat'
            
    if not isinstance(nii, ni.Nifti1Image):
        nii = ni.load(nii)

    affine = np.matrix(nii.get_affine()).getI()
    data = nii.get_data()

    # Do not mutate the original fiber-group. Instead, return a copy with the
    # transformation applied to it:
    fg = fg.xform(affine, inplace=False)
    
    stat_arr = np.empty(fg.n_fibers)
    for f_idx, fib in enumerate(fg.fibers):
        this_coord = ozu.nearest_coord(data, fib.coords[:, data_node])
        if this_coord is not None:
            stat_arr[f_idx] = data[this_coord]
        else: 
            stat_arr[f_idx] = None
    
    fg.fiber_stats[stat_name] = stat_arr

    return fg
    
def fg2volume(fg, stat, nii=None, shape=None, affine=None):
    """
    Take a statistic from a fiber-group and project it into a volume

    Parameters
    ----------
    fg: A FiberGroup class instance.

    stat: str
        The key into fg.fiber_stats to extract the statistic of interest.

    nii: str or nibabel.Nifti1Image class instance.
        A nifti file or nibabel Nifti1Image. If this is provided, the volume
        generated will have the same shape as nii and will inherit its affine
        (the affine will be applied to the fg before projecting the data into
        the volume)
 
    shape: If no nifti is provided, a shape is required The shape of the volume 

    affine: If no nifti is provided, an affine can still be provided as
        input. If no affine is provided, defaults to np.eye(4)

    """
    if nii is not None:
        if shape is not None or affine is not None:
            e_s = "Provide either nii input OR shape and affine, not both"
            raise ValueError(e_s)
        
        if not isinstance(nii, ni.Nifti1Image):
            # Get a Nifti1Image class instance:
            nii = ni.load(nii)

        shape = nii.get_shape()
        affine = np.matrix(nii.get_affine()).getI()

    else:
        if shape is None:
            e_s = "If no nii input is provided, must provide a shape"
            e_s += " for the volume."
            raise ValueError(e_s)
        if affine is None:
            affine = np.matrix(np.eye(4))

    stat_dict = fg.fiber_stats[stat]
    vol = np.zeros(shape)
    count_fibs = np.zeros(shape)    
    fg = fg.xform(affine, inplace=False)

    for fidx, fib in enumerate(fg.fibers):
        coords = fib.coords.astype(int)
        vol[coords[0], coords[1], coords[2]] += stat_dict[fidx]
        count_fibs[coords[0], coords[1], coords[2]] += 1

    # Put nans where there were no fibers:
    vol[np.where(count_fibs==0)] = np.nan
    vol /=count_fibs
        
    return vol


def resample_volume(source, target):
    """
    Resample the file in file_orig (full path string) to the resolution and
    affine of file_target (also a full path string)
    
    """
    if not isinstance(source, ni.Nifti1Image):
        source = ni.load(source)

    if not isinstance(target, ni.Nifti1Image):
        target = ni.load(target)

    source_vimg = as_volume_img(source)    
    # This does the magic - resamples using nearest neighbor interpolation:
    source_resamp_vimg = source_vimg.as_volume_img(affine=target.get_affine(),
                                         shape=target.shape[:3],
                                         interpolation='nearest')

    return ni.Nifti1Image(source_resamp_vimg.get_data(), target.get_affine())

