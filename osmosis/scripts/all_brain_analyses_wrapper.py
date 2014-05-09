"""

This is a wrapper for creating sge commands for parallel computation of model
parameters for SFM models from many subjects.

"""

import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge

cmd_file_path = "/home/klchan13/pycmd/"
python_path = "/home/klchan13/anaconda/bin/python"
bashcmd = "/home/klchan13/bashcmd.sh"
batch_sge_file = '/home/klchan13/batch_sge.sh'
hcp_path = '/hsgs/projects/wandell/klchan13/hcp_data_q3'
hostname = 'proclus.stanford.edu'
username = 'klchan13'
port = 22

# Analyses done:
# Reliability, isotropic model accuracy, diffusion model accuracy, fitted model parameters
ssh = sge.SSH(hostname=hostname,username=username, port=port)

batch_sge = []
sid_list = ["103414", "105115", "110411", "111312", "113619",
            "115320", "117122", "118730", "118932"]
for sid in sid_list:
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%sid)
    save_path = os.path.join(hcp_path, "%s"%sid)
    
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_registered.nii.gz"))
    wm_vox_num = np.sum(wm_data_file.get_data())
    for fODF in ["multi", "single"]:
        for im in ["bi_exp_rs", "single_exp_rs"]:
            if im == "bi_exp_rs":
                shorthand_im = "be"
            elif im == "single_exp_rs":
                shorthand_im = "se"
            # Reliability
            for i in np.arange(int(floor(wm_vox_num/2000.)+1)):
                import osmosis.parallel.emd_template as template
                qsub_cmd_gen(template, 'emd', i, sid, fODF, im, data_path, save_path)
            
            # Isotropic Model Accuracy
            for i in np.arange(int(floor(wm_vox_num/200) + 1)):
                import osmosis.parallel.im_accuracy_template as template
                qsub_cmd_gen(template, 'im_cod', i, sid, fODF, im, data_path, save_path)
                
            # Diffusion Model Accuracy
            for i in np.arange(int(floor(wm_vox_num/200)+1)):
                import osmosis.parallel.accuracy_template as template
                qsub_cmd_gen(template, 'sfm_cod', i, sid, fODF, im, data_path, save_path)
                    
            # Model Parameters
            for i in np.arange(int(floor(wm_vox_num/200)+1)):
                import osmosis.parallel.model_parameters_template as template
                qsub_cmd_gen(template, 'sfm_mp', i, sid, fODF, im, data_path, save_path)
        
# Add some header stuff:
batch_sge = ['#!/bin/bash'] + batch_sge
sge.write_file_ssh(ssh, batch_sge, batch_sge_file)

def qsub_cmd_gen(template, job_name, i, sid, fODF, im, data_path, save_path,
                 cmd_file_path=cmd_file_path, python_path=python_path,
                 bashcmd=bashcmd):
    reload(mb_template)
    template = sge.getsourcelines(mb_template)[0]

    params_dict = dict(i=i, sid=sid, fODF=fODF, im=im,
                       data_path=data_path, save_path=save_path)
    code = sge.add_params(template,params_dict)
    name = '%s_%s_%s%s'%(job_name,fODF,shorthand_im,i)
    cmd_file = os.path.join(cmd_file_path, '%s.py'%name)
    print("Generating: %s"%cmd_file)
                        
    sge.py_cmd(ssh,
            code,
            file_name=cmd_file,
            python=python_path)

    batch_sge.append(sge.qsub_cmd(
        '%s %s'%(bashcmd, cmd_file,name)))

