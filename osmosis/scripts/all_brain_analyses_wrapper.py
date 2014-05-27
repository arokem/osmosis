"""

This is a wrapper for creating sge commands for parallel computation of model
parameters for SFM models from many subjects.

"""

import os
import nibabel as nib
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge
import subprocess as sp
import time
import glob

cmd_file_path = "/home/klchan13/pycmd/"
python_path = "/home/klchan13/anaconda/bin/python"
bashcmd = "/home/klchan13/bashcmd.sh"
batch_sge_file = '/home/klchan13/batch_sge.sh'
hcp_path = '/hsgs/projects/wandell/klchan13/hcp_data_q3'
hostname = 'proclus.stanford.edu'
username = 'klchan13'
max_jobs = 8000.
port = 22

def qsub_cmd_gen(template, job_name, i, sid, fODF, im, data_path,
                 cmd_file_path=cmd_file_path, python_path=python_path,
                 bashcmd=bashcmd, mem=25):
    reload(template)
    template = sge.getsourcelines(template)[0]
    
    if job_name[0:2] != "im":
        params_dict = dict(i=i, sid=sid, fODF=fODF, im=im,
                        data_path=data_path)
        name = '%s_%s_%s%s'%(job_name,fODF,shorthand_im,i)
    else:
        params_dict = dict(i=i, sid=sid, im=im,
                        data_path=data_path)
        name = '%s_%s%s'%(job_name,shorthand_im,i)
        
    code = sge.add_params(template,params_dict)
    cmd_file = os.path.join(cmd_file_path, '%s.py'%name)
    print("Generating: %s"%cmd_file)
                        
    sge.py_cmd(ssh,
            code,
            file_name=cmd_file,
            python=python_path)
            
    cmd_file = os.path.join(cmd_file_path, '%s.py'%name)
    batch_sge.append(sge.qsub_cmd(
        '%s %s'%(bashcmd, cmd_file), name, mem_usage=mem))

# Analyses done:
# Reliability, isotropic model accuracy, diffusion model accuracy, fitted model parameters
ssh = sge.SSH(hostname=hostname,username=username, port=port)

batch_sge = []
sid_list = ["103414", "105115", "110411", "111312", "113619",
            "115320", "117122", "118730", "118932"]

# For aggregating later:
emd_file_names = []
other_file_names = []

# Start qsub generation:
subj_file_nums = []
for sid_idx, sid in enumerate(sid_list):
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%sid)
    
    wm_data_file = nib.load(os.path.join(data_path,"wm_mask_no_vent.nii.gz"))
    wm_vox_num = np.sum(wm_data_file.get_data())
    
    # For dividing up data
    emd_file_num = int(np.ceil(wm_vox_num/200.))
    others_file_num = int(np.ceil(wm_vox_num/2000.))
    subj_file_nums.append(np.array((emd_file_num, others_file_num)))
    for fODF in ["multi", "single"]:
        for im in ["bi_exp_rs", "single_exp_rs"]:
            if im == "bi_exp_rs":
                shorthand_im = "be"
            elif im == "single_exp_rs":
                shorthand_im = "se"
            
            # Reliability
            for i in np.arange(emd_file_num):
                import osmosis.parallel.emd_template as template
                if fODF == "single":
                    mem = 35
                else:
                    mem = 25
                    
                qsub_cmd_gen(template, 'emd_%s'%sid, i, sid, fODF, im, data_path, mem=mem)
                if sid_idx == 0:
                    emd_file_names.append("emd_%s_%s"%(fODF, shorthand_im))
            
            # Isotropic Model Accuracy
            # Only need to calculate isotropic models twice (one for each im) per
            # subject
            if fODF == "multi":
                for i in np.arange(others_file_num):
                    import osmosis.parallel.im_accuracy_template as template
                    qsub_cmd_gen(template, 'im_cod_%s'%sid, i, sid, fODF, im, data_path, mem=20)
                    if sid_idx == 0:
                        other_file_names.append("im_cod_%s"%shorthand_im)
                        other_file_names.append("im_predict_out_%s"%shorthand_im)
                        other_file_names.append("im_param_out_%s"%shorthand_im)
                    
            # Diffusion Model Accuracy
            for i in np.arange(others_file_num):
                import osmosis.parallel.accuracy_template as template
                if fODF == "single":
                    mem = 35
                else:
                    mem = 25
                    
                qsub_cmd_gen(template, 'sfm_cod_%s'%sid, i, sid, fODF, im, data_path, mem=mem)
                if sid_idx == 0:
                    other_file_names.append("sfm_predict_%s_%s"%(fODF,shorthand_im))
                    other_file_names.append("sfm_cod_%s_%s"%(fODF,shorthand_im))

            # Model Parameters
            for i in np.arange(others_file_num):
                import osmosis.parallel.model_params_template as template
                qsub_cmd_gen(template, 'sfm_mp_%s'%sid, i, sid, fODF, im, data_path, mem=25)
                if sid_idx == 0:
                    other_file_names.append("model_params_%s_%s"%(fODF, shorthand_im))
        
# Add some header stuff:
batch_sge = ['#!/bin/bash'] + batch_sge
sge.write_file_ssh(ssh, batch_sge, batch_sge_file)

# Read batch_sge.sh and get the individual qsub commands.
batch_sge = sp.check_output(["cat", "batch_sge.sh"])
cmd_line_split = batch_sge.split('\n')

# Check to see if the output files from each command exists already
# and eliminate them from the cmd_line
red_cmd_line = []
for cmd_idx in np.arange(1, len(cmd_line_split) - 1):
    pycmd = cmd_line_split[cmd_idx].split(' ')[2].split('_')
       
    if pycmd[0] == "im":
        sid_idx = len(pycmd) - 2
    else:
        sid_idx = len(pycmd) - 3
    
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%pycmd[sid_idx])
    
    # Rearrange the output file string
    file_str = []
    for str_splt in np.arange(len(pycmd)):
        if pycmd[1] != 'mp':
            if (str_splt != sid_idx) & (str_splt != len(pycmd) -1):
                file_str.append('%s_'%pycmd[str_splt])
            elif str_splt == len(pycmd) - 1:
                file_str.append('%s'%pycmd[str_splt])
        else:
            if str_splt == 0:
                file_str.append('model_params_')
                file_str.append('%s_'%pycmd[len(pycmd) - 2])
                file_str.append('%s'%pycmd[len(pycmd) - 1])
    file_str = ''.join(file_str)
    
    # Check if the file exists and if it doesn't, add to the new command list
    if glob.glob(os.path.join(data_path, '%s.*'%file_str)) == []:
        red_cmd_line.append(cmd_line_split[cmd_idx])

# For counting how many total qsub commands needed
count = len(red_cmd_line)

# Submit the first max number of jobs
for qsub_idx in np.arange(1, int(max_jobs+1)):
    cmd_arr = np.array(red_cmd_line[qsub_idx].split(' '))
    sp.call(list(cmd_arr[np.where(cmd_arr != '')]))
    
cur_job_num = len(str(sp.check_output(["qstat", "-u", "%s"%username])).split('\n'))
cur_submit_num = int(max_jobs + 1)

while cur_submit_num < count:
    while (cur_job_num == int(max_jobs)) | (cur_job_num > int(max_jobs)):
        time.sleep(10)
        cur_job_num = len(str(sp.check_output(["qstat", "-u", "%s"%username])).split('\n'))
    
    # If the number of jobs is less than the max number, submit a few more until
    # the max number of jobs is queued on the cluster
    num_to_submit = int(max_jobs) - cur_job_num
    qsub_range = np.arange(cur_submit_num, np.min([cur_submit_num + num_to_submit, count]))
    for qsub_idx in qsub_range:
        cur_submit_num = cur_submit_num + 1
        cmd_arr = np.array(red_cmd_line[qsub_idx].split(' '))
        sp.call(list(cmd_arr[np.where(cmd_arr != '')]))
        
    cur_job_num = len(str(sp.check_output(["qstat", "-u", "%s"%username])).split('\n'))

# Once the last of the jobs are submitted, keep checking to see if the jobs have finished
job_status = sp.check_output(["qstat", "-u", "%s"%username])
while '%s'%username in job_status:
    time.sleep(10)
    job_status = sp.check_output(["qstat", "-u", "%s"%username])
    
# Now that the jobs are done, aggregate files and reorganize.
for sid_idx, sid in enumerate(sid_list):
    data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%sid)
    os.chdir(data_path)
    wm_data_file = nib.load("wm_mask_no_vent.nii.gz")
    
    # Grab the number of files total for this subject
    emd_fnum = subj_file_nums[sid_idx][0]
    other_fnum = subj_file_nums[sid_idx][1]
    [missing_files_emd, vol_emd] = oio.place_files(emd_file_names, 200, emd_fnum,
                                                   wm_data_file, file_path = data_path,
                                                   save = True)
    [missing_files, vol] = oio.place_files(other_file_names, 2000, other_fnum,
                                           wm_data_file, file_path = data_path, save = True)
    # Keep a log of the missing files:
    str_to_write = ""
    for efname_idx, emd_fname in enumerate(emd_file_names):
        str_to_write = str_to_write + "%s is missing files:%s\n"%(emd_fname,
                                                                  missing_files_emd[efname_idx])        
    for fname_idx, fname in enumerate(other_file_names):
        str_to_write = str_to_write + "%s is missing files:%s\n"%(fname, missing_files[fname_idx])
    
    missing_files_txt = open("missing_files_%s.txt"%sid, "w")
    missing_files_txt.write("%s"%str_to_write)
    
    sp.call(['mkdir', 'file_pieces'])
    sp.call(['mkdir', 'analysis_results'])
    
    params = "mv aggre_* analysis_results | mv *.txt analysis_results | mv *.npy file_pieces"
    pipe = sp.Popen(params, shell=True)