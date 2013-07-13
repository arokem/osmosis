# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pymatbridge as pymat

# <codecell>

ip = get_ipython()
pymat.load_ipython_extension(ip, matlab='matlabr2010b')

# <codecell>

subject = 'FP'
if subject == 'HT':
    dt6 = '/biac2/wandell2/data/diffusion/takemura/20120725_3030/150dirs_b2000_2iso_2/dt6.mat'
elif subject == 'FP':
    dt6 = '/biac2/wandell2/data/diffusion/pestilli/20110922_1125/150dirs_b2000_2/dt6.mat'

# <codecell>

%%matlab -o cc_coords -i dt6
dt6 = dtiLoadDt6(dt6); 
cc_coords = dtiFindCallosum(dt6.dt6, dt6.b0, dt6.xformToAcpc); 

# <codecell>

cc_coords_mat = cc_coords.read() - 1  # Transform to 0-based indexing
cc_coords_mat.shape

# <codecell>

import nibabel as ni
import osmosis.volume as ozv
import osmosis.utils as ozu

# <codecell>

if subject == 'HT':
    dwi_ni = ni.load('/home/arokem/data/osmosis/HT/0012_01_DTI_2mm_b2000_150dir_aligned_trilin.nii.gz')
    t1_ni = ni.load('/home/arokem/anatomy/takemura/t1.nii.gz')
if subject == 'FP':
    dwi_ni = ni.load('/home/arokem/data/osmosis/FP/0005_01_DTI_2mm_150dir_2x_b2000_aligned_trilin.nii.gz')
    t1_ni = ni.load('/home/arokem/anatomy/pestilli//t1.nii.gz')

# <codecell>

vol = ozu.nans(dwi_ni.shape[:3])

# <codecell>

cc_coords_xform = ozu.xform(cc_coords_mat, np.matrix(dwi_ni.get_affine()).getI())

# <codecell>

vol[cc_coords_xform[0,:].astype(int), cc_coords_xform[1,:].astype(int), cc_coords_xform[2,:].astype(int)] = 1

# <codecell>

new_ni = ni.Nifti1Image(vol, dwi_ni.get_affine())

# <codecell>

new_ni.to_filename('/home/arokem/data/osmosis/%s/%s_cc.nii.gz'%(subject, subject))

