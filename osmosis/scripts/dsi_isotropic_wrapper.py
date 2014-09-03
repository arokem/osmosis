"""

This is a wrapper for creating sge commands for parallel computation of model
cross-prediction rRMSE for SFM models.

This needs to be run on proclus and after running ssd_wrapper. Instructions are
essentially the same as those mentioned in there. 
"""

import os
import nibabel as nib
import numpy as np
from osmosis.parallel import sge

import osmosis.parallel.dsi_isotropic_template as template
reload(template)
template = sge.getsourcelines(template)[0]

ssh = sge.SSH(hostname='proclus.stanford.edu',username='arokem', port=22)


data_path = data_path = '/biac4/wandell/data/qytian/DSIProject/'
DSI515_mask = nib.load(data_path +
                           '/DSI515/mask_mask_hand.nii.gz').get_data()

vox_per_job = 10
batch_sge = []
for i in xrange(int(np.ceil(np.sum(DSI515_mask)/vox_per_job))):
    
    params_dict =  dict(i=i)
    code = sge.add_params(template, params_dict)
    name ='dsi_iso_kfold%s'%i
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


