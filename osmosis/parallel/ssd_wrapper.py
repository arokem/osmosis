import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge
import ssd_template
reload(ssd_template)
template = sge.getsourcelines(ssd_template)[0]
alphas = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05]
rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

data_path = '/biac4/wandell/biac2/wandell6/data/arokem/osmosis/'

credentials = sge._get_credentials(hostname='proclus.stanford.edu',
                                   username='arokem')

batch_sge = []
for subject in ['FP', 'HT']:
    subject_path = os.path.join(oio.data_path, subject)
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    wm_nifti = ni.load(wm_mask_file)
    wm_data = wm_nifti.get_data()
    n_wm_vox = np.sum(wm_data)
    for b in [1000, 2000, 4000]:
        ad_rd = oio.get_ad_rd(subject, b)
        for data_i, data in enumerate(oio.get_dwi_data(b, subject)):
            file_stem = (data_path + '%s/'%subject +
                         data[0].split('/')[-1].split('.')[0])
            for rho in rhos:
                for alpha in alphas:
                    for i in range(int(n_wm_vox/1000)+2):
                        params_dict =  dict(
                            data_path=data_path,
                            i=i,
                            data_i=data_i,
                            subject=subject,
                            b=b,
                            rho=rho,
                            alpha=alpha,
                            ad=ad_rd[data_i]['AD'],
                            rd=ad_rd[data_i]['RD'],
                            wm_file = "%s_wm_mask.nii.gz"%subject,
                    params_file = "%s_SSD_rho%s_alpha%s_%03d.nii.gz"%(file_stem,
                                                            rho,
                                                            alpha,
                                                            i))
                                    
                        code = sge.add_params(template,params_dict)
                        name = 'ssd_%s_b%s_data%s_rho%s_alpha%s_%03d'%(
                            subject, b, data_i+1, rho, alpha, i)
                        cmd_file = '/home/arokem/pycmd/%s.py'%name
                        print("Generating: %s"%cmd_file)

                        sge.py_cmd(code,
                                   hostname = credentials[0],
                                   username = credentials[1],
                                   password = credentials[2],
                                   python='/home/arokem/anaconda/bin/python',
                                   cmd_file=cmd_file)
                                                 
                        batch_sge.append(sge.qsub_cmd('bashcmd.sh /home/arokem/pycmd/%s.py'%name, name))

# Add some header stuff:
#batch_sge = ['export PATH=$PATH:/hsgs/software/oge2011.11p1/bin/linux-x64/'] + batch_sge
#batch_sge = ['export SGE_ROOT=/hsgs/software/oge2011.11p1'] + batch_sge
batch_sge = ['#!/bin/bash'] + batch_sge

sge.write_file_ssh(batch_sge, 'batch_sge.sh',
                   hostname = credentials[0],
                   username = credentials[1],
                   password = credentials[2])

    #stdin, stdout, stderr = sge.ssh('./batch_sge.sh',
    #                           hostname = credentials[0],
    #                           username = credentials[1],
    #                           password = credentials[2])
