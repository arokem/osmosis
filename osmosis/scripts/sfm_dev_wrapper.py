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

dwi_dirs =\
    [
    '/biac4/wandell/biac2/wandell2/data/WH/001_RM/DTI/20110727_0755/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/002_KM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/003_SP/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/004_JY/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/005_KK/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/006_FP/DTI/dti150ls',
    '/biac4/wandell/biac2/wandell2/data/WH/007_JW/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/008_AM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/009_AL/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/010_TD/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/011_HO/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/012_JR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/013_KH/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/014_HJ/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/015_CS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/016_CC/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/017_LB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/018_KK/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/019_LH/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/020_LY/DTI/dti30ls',
    '/biac4/wandell/biac2/wandell2/data/WH/021_DP/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/022_MP/DTI/dti30ls',
    '/biac4/wandell/biac2/wandell2/data/WH/023_MG/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/024_RK/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/025_NA/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/026_JW/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/027_CS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/028_MJ/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/029_JA/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/030_BA/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/031_MP/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/032_NL/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/033_SB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/035_BB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/036_HM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/037_JF/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/038_BW/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/039_KR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/040_NR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/041_LL/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/042_CB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/043_TR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/044_JD/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/045_BS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/046_BB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/047_FG/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/048_HG/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/049_SL/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/050_JL/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/051_DF/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/052_NA/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/053_VV/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/054_JR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/055_CS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/056_JS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/057_CS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/058_AW/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/059_MR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/060_MB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/061_DS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/062_ES/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/063_IS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/064_EG/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/065_HP/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/066_AS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/067_NS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/068_JM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/069_HH/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/070_SK/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/071_JK/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/072_EH/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/073_SVB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/074_MVB/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/075_LV/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/076_CR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/077_GR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/078_BTL/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/079_MBH/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/080_DR/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/082_JW/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/081_PS/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/083_VAG/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/084_JAG/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/085_RD/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/086_VD/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/087_HM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/088_RM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/089_SM/DTI/dti96ls',
    '/biac4/wandell/biac2/wandell2/data/WH/052_NA/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/053_VV/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/054_JR/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/043_TR/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/055_CS/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/058_AW/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/059_MR/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/060_MB/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/064_EG/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/065_HP/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/066_AS/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/068_JM/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/070_SK/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/071_JK/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/075_LV/DTI/dti96_outExclude',
    '/biac4/wandell/biac2/wandell2/data/WH/076_CR/DTI/dti96_outExclude',
    ]

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
