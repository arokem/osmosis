"""

This is a wrapper for creating sge commands for parallel computation of model
cross-prediction rRMSE for SFM models.

This needs to be run on proclus and after running ssd_wrapper. Instructions are
essentially the same as those mentioned in there. 
"""

import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge

import osmosis.parallel.ssd_predict_template as template
reload(template)
template = sge.getsourcelines(template)[0]

alphas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
l1_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

data_path = '/hsgs/u/arokem/tmp/'

ssh = sge.SSH(hostname='proclus.stanford.edu',username='arokem', port=22)

batch_sge = []
for subject in ['SUB1']: #,'HT']    
    wm_file = "%s_wm_mask.nii.gz"%subject
    for b in [1000, 2000, 4000]:
        data = oio.get_dwi_data(b, subject)
        file_stems = [data_path + '%s/'%subject +
                      d[0].split('/')[-1].split('.')[0] for d in data]
        for l1_ratio in l1_ratios:
            for alpha in alphas:
                
                params_file1 ="%s_SSD_l1ratio%s_alpha%s.nii.gz"%(
                    file_stems[0],
                    l1_ratio, alpha)


                params_file2="%s_SSD_l1ratio%s_alpha%s.nii.gz"%(
                    file_stems[1],
                    l1_ratio, alpha)

                params_dict =  dict(
                    data_path=data_path,
                    subject=subject,
                    b=b,
                    l1_ratio=l1_ratio,
                    alpha=alpha,
                    wm_file=wm_file,
                    params_file1=params_file1,
                    params_file2=params_file2)
                
                code = sge.add_params(template, params_dict)
                name ='ssd_predict%s_b%s_l1ratio%s_alpha%s'%(
                    subject, b, l1_ratio, alpha)
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
#batch_sge = ['export PATH=$PATH:/hsgs/software/oge2011.11p1/bin/linux-x64/'] + batch_sge
#batch_sge = ['export SGE_ROOT=/hsgs/software/oge2011.11p1'] + batch_sge
batch_sge = ['#!/bin/bash'] + batch_sge
sge.write_file_ssh(ssh, batch_sge, '/home/arokem/batch_sge.sh')


#stat = os.system('scp -c blowfish -C tmp/* %s:~/pycmd/.'%ssh.hostname)
#if stat != 0:
#    print "what what!"

#ssh.exec_command('./batch_sge.sh')
#ssh.disconnect()
