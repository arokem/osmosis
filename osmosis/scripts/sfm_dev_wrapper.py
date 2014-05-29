"""

This is a wrapper for creating sge commands for parallel computation of model
parameters for SFM models from many subjects.

"""

import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge

import osmosis.parallel.sfm_dev_template as ssd_template
reload(ssd_template)
template = sge.getsourcelines(ssd_template)[0]


ssh = sge.SSH(hostname='proclus.stanford.edu',username='arokem', port=22)
ssh.connect()

dwi_dirs =\
    ['/biac4/wandell/biac2/wandell2/data/WH/001_RM/DTI/raw/11_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/002_KM/DTI/raw/0014_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/003_SP/DTI/raw/0014_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/004_JY/DTI/raw/15_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/005_KK/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/006_FP/DTI/raw/20110922_1125/0005_01_DTI_2mm_150dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/007_JW/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/008_AM/DTI/raw/0014_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/009_AL/DTI/raw/0013_01_DTI_2mm_96dir_b2000_Accel_25xmin_full_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/010_TD/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/011_HO/DTI/raw/0014_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/012_JR/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/013_KH/DTI/raw/14_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/014_HJ/DTI/raw/0008_01_DTI_2mm_96dir_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/015_CS/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/016_CC/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/017_LB/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/018_KK/DTI/raw/16_1_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/019_LH/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/021_DP/DTI/raw/16_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/023_MG/DTI/raw/18_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/024_RK/DTI/raw/0014_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/025_NA/DTI/raw/17_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/026_JW/DTI/raw/13_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/027_CS/DTI/raw/13_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/028_MJ/DTI/raw/12_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/029_JA/DTI/raw/15_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/030_BA/DTI/raw/18_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/031_MP/DTI/raw/12_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/032_NL/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/033_SB/DTI/raw/16_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/035_BB/DTI/raw/15_DTI_2mm_96dir_2x_b2000_1_0_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/036_HM/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/037_JF/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/038_BW/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/039_KR/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/040_NR/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/041_LL/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/042_CB/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/043_TR/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/044_JD/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/045_BS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/046_BB/DTI/raw/0020_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/047_FG/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/048_HG/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/049_SL/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/050_JL/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/051_DF/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/052_NA/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/053_VV/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/054_JR/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/055_CS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/056_JS/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/057_CS/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/058_AW/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/059_MR/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/060_MB/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/061_DS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/062_ES/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/063_IS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/064_EG/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/065_HP/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/066_AS/DTI/raw/0018_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/067_NS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/068_JM/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/069_HH/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/070_SK/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/071_JK/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/072_EH/DTI/raw/0017_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/073_SVB/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/074_MVB/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/075_LV/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/076_CR/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/077_GR/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/078_BTL/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/079_MBH/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/080_DR/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/082_JW/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/081_PS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/083_VAG/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/084_JAG/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/085_RD/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/086_VD/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/087_HM/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/088_RM/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/089_SM/DTI/raw/0013_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/090_EM/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/091_RD/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/092_LP/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/093_LB/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/094_MN/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/095_BP/DTI/raw/0016_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/096_RS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/097_CZ/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/098_DN/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/099_AD/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/100_WK/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/101_JG/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/102_LS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/103_LD/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/105_HF/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/106_DS/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/107_KC/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin',
'/biac4/wandell/biac2/wandell2/data/WH/108_BG/DTI/raw/0015_01_DTI_2mm_96dir_2x_b2000_aligned_trilin']

batch_sge = []
for this_dir in dwi_dirs:
    params_dict =  dict(this_dir=this_dir)
    code = sge.add_params(template,params_dict)
    name = 'sfm_dev_%s'%(this_dir.split('/')[7])
    cmd_file = '/home/arokem/pycmd/%s.py'%name
    print("Generating: %s"%cmd_file)

    sge.py_cmd(ssh,
               code,
               file_name=cmd_file,
               python='/home/arokem/anaconda/bin/python')

    cmd_file = '/home/arokem/pycmd/%s.py'%name
    batch_sge.append(sge.qsub_cmd(
        '/home/arokem/bashcmd.sh %s'%cmd_file,name))

# Add some header stuff:
batch_sge = ['#!/bin/bash'] + batch_sge
sge.write_file_ssh(ssh, batch_sge, '/home/arokem/batch_sge.sh')
