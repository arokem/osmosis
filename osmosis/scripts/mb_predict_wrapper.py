"""

This is a wrapper for creating sge commands for parallel computation of model
parameters for SFM models from many subjects.

"""

import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge

import osmosis.parallel.mb_predict_template as mb_template
reload(mb_template)
template = sge.getsourcelines(mb_template)[0]


ssh = sge.SSH(hostname='proclus.stanford.edu',username='klchan13', port=22)

batch_sge = []
for i in range(65): 
    params_dict = dict(i=i)
    code = sge.add_params(template,params_dict)
    name = 'mb_predict_em%s'%(i)
    cmd_file = '/home/klchan13/pycmd/%s.py'%name
    print("Generating: %s"%cmd_file)
                        
    sge.py_cmd(ssh,
               code,
               file_name=cmd_file,
               python='/home/klchan13/anaconda/bin/python')

    cmd_file = '/home/klchan13/pycmd/%s.py'%name
    batch_sge.append(sge.qsub_cmd(
        '/home/klchan13/bashcmd.sh %s'%cmd_file,name))

# Add some header stuff:
batch_sge = ['#!/bin/bash'] + batch_sge
sge.write_file_ssh(ssh, batch_sge, '/home/klchan13/batch_sge.sh')

