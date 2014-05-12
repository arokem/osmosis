"""

This is a wrapper for creating sge commands for parallel computation of model
parameters for SFM models.

This needs to be run on proclus:

1. Log into proclus
2. Make sure that the output directories are empty:
   /hsgs/u/arokem/tmp
   ~/sgeoutput
   ~/pycmd
   ~/batch_sge.sh

3. Run:  
   python ssd_wrapper.py
4. This should create a new batch_sge.sh file in your home directory
5. Run it:
   ./batch_sge.sh

6. Now you need to wait for all the processes to go through and for the nifti
parameter files to be created.

7. Once that's done, you should be able to run:
   python ssd_reassmble
Which will generate the parameter files. Use these. 

"""

import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge

import osmosis.parallel.ssd_template as ssd_template
reload(ssd_template)
template = sge.getsourcelines(ssd_template)[0]

alphas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
l1_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

data_path = '/hsgs/u/arokem/tmp/'

ssh = sge.SSH(hostname='proclus.stanford.edu',username='arokem', port=22)

batch_sge = []
for subject in ['SUB1']:#,'SUB2']:
    subject_path = os.path.join(oio.data_path, subject)
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    wm_nifti = ni.load(wm_mask_file)
    wm_data = wm_nifti.get_data()
    n_wm_vox = np.sum(wm_data)
    
    wm_file = "%s_wm_mask.nii.gz"%subject
    for b in [1000, 2000, 4000]:
        ad_rd = oio.get_ad_rd(subject, b)
        for data_i, data in enumerate(oio.get_dwi_data(b, subject)):
            file_stem = (data_path + '%s/'%subject +
                         data[0].split('/')[-1].split('.')[0])
            for l1_ratio in l1_ratios:
                for alpha in alphas:
                    for i in range(int(n_wm_vox/10000)+2):
                        params_file="%s_SSD_l1ratio%s_alpha%s_%03d.nii.gz"%(
                            file_stem,
                            l1_ratio,
                            alpha,
                            i)

                        params_dict =  dict(
                            data_path=data_path,
                            i=i,
                            data_i=data_i,
                            subject=subject,
                            b=b,
                            l1_ratio=l1_ratio,
                            alpha=alpha,
                            ad=ad_rd[data_i]['AD'],
                            rd=ad_rd[data_i]['RD'],
                            wm_file=wm_file,
                            params_file = params_file)
                                    
                        code = sge.add_params(template,params_dict)
                        name = 'ssd_%s_b%s_data%s_l1ratio%s_alpha%s_%03d'%(
                            subject, b, data_i+1, l1_ratio, alpha, i)
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
