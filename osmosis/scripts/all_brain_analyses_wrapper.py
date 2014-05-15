"""

This is a wrapper for creating sge commands for parallel computation of model
parameters for SFM models from many subjects.

"""

import os
import nibabel as ni
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge
import subprocess as sp
import time

cmd_file_path = "/home/klchan13/pycmd/"
python_path = "/home/klchan13/anaconda/bin/python"
bashcmd = "/home/klchan13/bashcmd.sh"
batch_sge_file = '/home/klchan13/batch_sge.sh'
hcp_path = '/hsgs/projects/wandell/klchan13/hcp_data_q3'
hostname = 'proclus.stanford.edu'
username = 'klchan13'
max_jobs = 8000.
port = 22

# Analyses done:
# Reliability, isotropic model accuracy, diffusion model accuracy, fitted model parameters
ssh = sge.SSH(hostname=hostname,username=username, port=port)

batch_sge = []
sid_list = ["103414", "105115", "110411", "111312", "113619",
            "115320", "117122", "118730", "118932"]
count = 0 # For counting how many total qsub commands needed

# For aggregating later:
emd_file_names = []
other_file_names = []

# Start qsub generation:
subj_file_nums = []
for sid_idx, sid in enumerate(sid_list):
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%sid)
    
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_no_vent.nii.gz"))
    wm_vox_num = np.sum(wm_data_file.get_data())
    for fODF in ["multi", "single"]:
        for im in ["bi_exp_rs", "single_exp_rs"]:
            if im == "bi_exp_rs":
                shorthand_im = "be"
            elif im == "single_exp_rs":
                shorthand_im = "se"
            
            # For dividing up data
            emd_file_num = int(np.ceil(wm_vox_num/200.)
            others_file_num = int(np.ceil(wm_vox_num/2000.)
            subj_file_nums.append(np.array((emd_file_num, others_file_num)))
            
            # Reliability
            for i in np.arange(emd_file_num):
                import osmosis.parallel.emd_template as template
                qsub_cmd_gen(template, 'emd', i, sid, fODF, im, data_path)
                count = count + 1
                if sid_idx == 0:
                    emd_file_names.append("emd_%s_%s"%(fODF, shorthand_im))
            
            # Isotropic Model Accuracy
            # Only need to calculate isotropic models twice (one for each im) per
            # subject
            if fODF == "multi":
                for i in np.arange(others_file_num):
                    import osmosis.parallel.im_accuracy_template as template
                    qsub_cmd_gen(template, 'im_cod', i, sid, fODF, im, data_path)
                    count = count + 1
                    if sid_idx == 0:
                        other_file_names.append("im_cod_%s"%shorthand_im)
                        other_file_names.append("im_predict_out_%s"%shorthand_im)
                        other_file_names.append("im_param_out_%s"%shorthand_im)
                    
            # Diffusion Model Accuracy
            for i in np.arange(others_file_num):
                import osmosis.parallel.accuracy_template as template
                qsub_cmd_gen(template, 'sfm_cod', i, sid, fODF, im, data_path)
                count = count + 1
                if sid_idx == 0:
                    other_file_names.append("sfm_predict_%s_%s"%(fODF,shorthand_im))
                    other_file_names.append("sfm_cod_%s_%s"%(fODF,shorthand_im))
                    
            # Model Parameters
            for i in np.arange(others_file_num):
                import osmosis.parallel.model_parameters_template as template
                qsub_cmd_gen(template, 'sfm_mp', i, sid, fODF, im, data_path)
                count = count + 1
                if sid_idx == 0:
                    other_file_names.append("model_params_%s_%s"%(fODF, shorthand_im))
        
# Add some header stuff:
batch_sge = ['#!/bin/bash'] + batch_sge
sge.write_file_ssh(ssh, batch_sge, batch_sge_file)


# Stagger the jobs so you don't go over the max limit
job_status = sp.check_output(["qstat", "-u", "%s"%username])
batch_sge = sp.check_output(["cat", "batch_sge.sh"])
cmd_line_split = batch_sge.split('\n')
total_submits = int(np.ceil(count/max_jobs))

for jobs_round in np.arange(total_submits):
    if jobs_round == 0:
        low = 1
    else:
        low = jobs_round*max_jobs
        
    high = np.min([(i+1)*max_jobs, int(count)])
    while "%s"%username in job_status:
        time.sleep(10)
        job_status = sp.check_output(["qstat", "-u", "%s"%username])
    
    for qsub_idx in np.arange(low, high):
        cmd_arr = np.array(cmd_line_split[qsub_idx].split(' '))
        sp.call(list(cmd_arr[np.where(cmd_arr != '')]))

# Aggregate files and reorganize.
for sid_idx, sid in enumerate(sid_list):
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%sid)
    os.chdir(data_path)
    wm_data_file = nib.load("wm_mask_no_vent.nii.gz")
    
    # Grab the number of files total for this subject
    emd_fnum = subj_file_nums[sid_idx][0]
    other_fnum = subj_file_nums[sid_idx][1]
    [missing_files_emd, vol_emd] = oio.place_files(emd_file_names, 200, emd_fnum,
                                                   wm_data_file, save = "Yes")
    [missing_files, vol] = oio.place_files(other_file_names, 2000, other_fnum,
                                           wm_data_file, save = "Yes")
    # Keep a log of the missing files:
    missing_files_txt = open("missing_files_%s.txt"%sid, "w")
    missing_files_txt.write("%s\n"%sid)
    for efname_idx, emd_fname in enumerate(emd_file_names):
        missing_files_txt.write("%s is missing files:%s\n"%(emd_fname, missing_files_emd[efname_idx]))
        
    for fname_idx, fname in enumerate(other_file_names):
        missing_files_txt.write("%s is missing files:%s\n"%(emd_fname, missing_files[fname_idx]))
    
    missing_files_txt.close()
    
    sp.call(['mkdir', 'file_pieces'])
    sp.call(['mkdir', 'analysis_results'])
    sp.call(['mv', 'aggre_*', 'analysis_results'])
    sp.call(['mv', '*.txt', 'analysis_results'])

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