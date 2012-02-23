"""
microtrack.volume
=================

Integration of data from volumes into fiber groups


"""
import os

import numpy as np
import nibabel as ni

import microtrack.fibers as mtf
import microtrack.io as io
import microtrack.utils as mtu


def nii2fg(fg, nii, data_node=0, stat_name=None):
    """
    Attach data from a nifti volume to the fiber-group fiber_stats dict

    fg: A microtrack.fibers.FiberGroup class instance or a full path to a pdb
        file.

    nii: A nibabel.Nifti1 class instance or a full path to a nifti file.

    data_node: Whether to use data from the beginning (0) or end (-1) of each
    fiber as the
    
    """

    if data_node not in [0,-1]:
        e_s = "Can only attach data from the beginning or end of the fiber"
        raise ValueError(e_s)
    
    if stat_name is None:
        if isinstance(nii, str):
            stat_name = os.path.split(nii)[-1]
        else:
            stat_name = 'stat'
            
    if not isinstance(nii, ni.Nifti1Image):
        nii = ni.load(nii)

    if not isinstance(fg, mtf.FiberGroup):
        fg = io.fg_from_pdb(fg)

    affine = nii.get_affine()
    data = nii.get_data()

    # Do not mutate the original fiber-group. Instead, return a copy with the
    # transformation applied to it:
    fg = fg.xform(np.matrix(affine).getI(), inplace=False)
    
    stat_arr = np.empty(fg.n_fibers)
    for f_idx, fib in enumerate(fg.fibers):
        this_coord = mtu.nearest_coord(data, fib.coords[:, data_node])
        if this_coord is not None:
            stat_arr[f_idx] = data[this_coord]
        else: 
            stat_arr[f_idx] = None
    
    fg.fiber_stats[stat_name] = stat_arr

    return fg
    
