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

data_path = '/tmp/arokem/data/osmosis'#'/biac4/wandell/biac2/wandell6/data/arokem/osmosis/'

ssh = sge.SSH(hostname='proclus.stanford.edu',username='arokem', port=22)

batch_sge = []
for subject in ['FP']:#, 'HT']:
    subject_path = os.path.join(oio.data_path, subject)
    wm_mask_file = os.path.join(subject_path, '%s_wm_mask.nii.gz'%subject)
    wm_nifti = ni.load(wm_mask_file)
    wm_data = wm_nifti.get_data()
    n_wm_vox = np.sum(wm_data)
    
    wm_file = "%s_wm_mask.nii.gz"%subject
    for b in [1000, 2000, 4000]:
        ad_rd = oio.get_ad_rd(subject, b)
        for data_i, data in enumerate(oio.get_dwi_data(b, subject)):
            file_stem = ('/tmp/arokem/data/osmosis' + '%s/'%subject +
                         data[0].split('/')[-1].split('.')[0])
            for rho in rhos:
                for alpha in alphas:
                    for i in range(int(n_wm_vox/10000)+2):
                        params_file="%s_SSD_rho%s_alpha%s_%03d.nii.gz"%(
                            file_stem,
                            rho,
                            alpha,
                            i)

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
                            wm_file=wm_file,
                            params_file = params_file)
                                    
                        code = sge.add_params(template,params_dict)
                        name = 'ssd_%s_b%s_data%s_rho%s_alpha%s_%03d'%(
                            subject, b, data_i+1, rho, alpha, i)
                        cmd_file = 'tmp/%s.py'%name
                        print("Generating: %s"%cmd_file)
                        
                        sge.py_cmd(ssh,
                                   code,
                                   file_name=cmd_file,
                                   python='/home/arokem/anaconda/bin/python')

                        cmd_file = '/home/arokem/pycmd/%s.py'%name
                        batch_sge.append(sge.qsub_cmd('bashcmd.sh %s'%cmd_file,
                                                      name))

# Add some header stuff:
#batch_sge = ['export PATH=$PATH:/hsgs/software/oge2011.11p1/bin/linux-x64/'] + batch_sge
#batch_sge = ['export SGE_ROOT=/hsgs/software/oge2011.11p1'] + batch_sge
batch_sge = ['#!/bin/bash'] + batch_sge

sge.write_file_ssh(ssh, batch_sge, 'tmp/batch_sge.sh')

stat = os.system('scp -c blowfish -C tmp/* %s:~/pycmd/.'%ssh.hostname)
if stat != 0:
    print "what what!"

#ssh.exec_command('./batch_sge.sh')
#ssh.disconnect()
