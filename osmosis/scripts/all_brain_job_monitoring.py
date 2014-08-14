import os
import nibabel as nib
import numpy as np
import osmosis.io as oio
from osmosis.parallel import sge
import subprocess as sp
import time
import glob
import sys

sid_list = ["103414", "105115", "110411", "111312", "113619",
            "115320", "117122", "118730", "118932"]
hcp_path = '/hsgs/projects/wandell/klchan13/hcp_data_q3'
username = 'klchan13'
batch_sge = sp.check_output(["cat", "batch_sge.sh"])
cmd_line_split = batch_sge.split('\n')

# Check to see if the output files from each command exists already
# and eliminate them from the cmd_line
job_status = sp.check_output(["qstat", "-u", "%s"%username])
while '%s'%username in job_status:
    time.sleep(10)
    red_cmd_line = []
    for cmd_idx in np.arange(2, len(cmd_line_split)-1):
        pycmd = cmd_line_split[cmd_idx].split(' ')[2].split('_')

        if pycmd[0] == "im":
            sid_idx = len(pycmd) - 2
        else:
            sid_idx = len(pycmd) - 3
        data_path = os.path.join(hcp_path, "%s/T1w/Diffusion"%pycmd[sid_idx])
        file_str_list = []
        for str_splt in np.arange(len(pycmd)):
            if pycmd[1] != 'mp':
                if (str_splt != sid_idx) & (str_splt != len(pycmd) -1):
                    file_str_list.append('%s_'%pycmd[str_splt])
                elif str_splt == len(pycmd) - 1:
                    file_str_list.append('%s'%pycmd[str_splt])
            else:
                if str_splt == 0:
                    file_str_list.append('model_params_')
                    file_str_list.append('%s_'%pycmd[len(pycmd) - 2])
                    file_str_list.append('%s'%pycmd[len(pycmd) - 1])
        file_str = ''.join(file_str_list)
        if glob.glob(os.path.join(data_path, '%s.*'%file_str)) == []:
            red_cmd_line.append(cmd_line_split[cmd_idx])

    submitted_jobs = str(sp.check_output(["qstat", "-u",
                            "%s"%username])).split('\n')

    # Find the jobs with "r" in them which indicates that it's currently using
    # a node.
    cur_nodes = 0
    for sj in submitted_jobs:
        if '     r    ' in sj:
            cur_nodes = cur_nodes + 1

    # Print out how many jobs are done and how many nodes are currently being
    # used.
    sys.stdout.write('\r' + "%s out of %s jobs done.  Using %s nodes."%(
                     len(cmd_line_split) - int(len(red_cmd_line)),
                     len(cmd_line_split), cur_nodes))
    sys.stdout.flush()
    job_status = sp.check_output(["qstat", "-u", "%s"%username])
