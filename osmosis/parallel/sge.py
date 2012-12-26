"""

Submissions on an SGE, over ssh

Warning: some of this will not work in IPython for the time being, until
getpass issues are resolved.

"""

import os
import getpass
import inspect
import traceback

# This does ssh:
import paramiko

def getsourcelines(object):
    """Return a list of source lines and starting line number for an object.

    Parameters
    ----------
    object : a python object 
       The argument may be a module, class, method, function, traceback, frame,
       or code object.  The source code is returned as a string with the lines
       corresponding to the object and the line number indicates where in the
       original source file the first line of code was found.  An IOError is
       raised if the source code cannot be retrieved."""
    
    lines, lnum = inspect.findsource(object)
    if inspect.ismodule(object):
        lnum = 0
        ss = ''
        for x in lines:
            ss += x
    else:
        lines = inspect.getblock(lines[lnum:])
        lnum = lnum + 1
        ss = ''
        for x in lines:
            ss += x

    return ss, 0
        

def _get_credentials(hostname=None, username=None, password=None):
    # Get your hostname and credentials:
    if hostname is None:
        hostname = raw_input('Hostname: ')

    if username is None:
        default_username = getpass.getuser()
        username = raw_input('Username [%s]: ' % default_username)
        if len(username) == 0:
            username = default_username

    # We need to check whether we are in an ipython session, to know how to get
    # the password:
    if password is None: 
        try:
           ip = get_ipython()
           is_ip = True
        except NameError: 
           is_ip = False
        if is_ip:
           password = raw_input('Password for %s@%s: ' % (username, hostname))
        else:
           password = getpass.getpass('Password for %s@%s: ' % (username,
                                                                hostname))

    return hostname, username, password

def ssh(command, hostname=None, username=None, password=None, port=22):
    """
    Send a command over ssh to a host, stdout will be posted back to your local
    terminal 
    """
    hostname, username, password = _get_credentials(hostname, username, password)
    try:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy)
        client.connect(hostname, port, username, password)
        stdin, out, err = client.exec_command(command)
        stdout = ''
        for line in out:
            stdout += line

        stderr = ''
        for line in err:
            stderr += line

        client.close()
    
    except Exception, e:
        print '*** Caught exception: %s: %s' % (e.__class__, e)
        traceback.print_exc()
        try:
            client.close()
        except:
            pass
        stdin = False
        stout = False
        stderr = False

    return stdin, stdout, stderr

def add_params(s, params_dict):
    """

    Add parameter values from a dict to a code string. This sets the values of
    particular elements of the code, so that many different versions of the
    same code can be run with different parameter settings

    """
    for k,v in params_dict.items():
       if isinstance(v, str):
          s = '%s="%s"\n'%(k,v) + s
       else: 
          s = "%s = %s\n"%(k,v) + s

    return s

def python_call(cmd):
    """
    Assemble a call to python to execute a command string (This can probably be
    rather long thing, if you want)
    """
    return "python -c'%s'"%cmd

def qsub_cmd(call, name, working_dir='cwd', shell='/bin/bash',
             email='$USER@stanford.edu', mem_usage=4, priority=0,
             flags='', output_dir='sgeoutput'):
    """
    This puts together the qsub command.

    Parameters
    ----------
    The call
    
    
    """
    return "qsub -N %s -o %s -e %s -l h_vmem=%sg -p %d %s -S %s %s"%(name,
                                                                     output_dir,
                                                                     output_dir,
                                                                     mem_usage,
                                                                     priority,
                                                                     flags,
                                                                     shell,
                                                                     call)


def py_cmd(cmd, hostname=None, username=None, password=None, python=None,
           cmd_file='~/pycmd/cmd.py'):
    """

    This generates the python script on the cluster.
    
    """
    hostname, username, password = _get_credentials(hostname, username, password)
    if python is None:
       python = ssh('which python', hostname, username, password)[1].strip('\n')
    #print python
    hdr_line = '#!%s'%python

    ssh('touch %s'%cmd_file, hostname, username, password)
    ssh("echo '%s' > %s"%(hdr_line, cmd_file), hostname, username, password)

    # Make sure quotes get correctly escaped in:
    cmd.replace('"', '\"')
    cmd.replace("'", "\'")
    
    ssh("echo '%s' >> %s"%(cmd, cmd_file), hostname, username, password)
    stdin, stdout, stderr = ssh('chmod 755 %s'%cmd_file, hostname, username,
                                password)

def write_file_ssh(line_list, file_name, hostname=None, username=None,
                   password=None, executable=True):
   """
   Write a file made out of the lines in line_list on a remote machine

   Note
   ----
   This will clobber existing files! Need to figure out a way to be more
   careful with that.
   """
   hostname, username, password = _get_credentials(hostname, username, password)
   ssh('touch %s'%file_name, hostname, username, password)
   s = ''

   for line in line_list:
      s = s + line + '\n'

   ssh("echo '%s' >> %s"%(s, file_name), hostname, username, password ) 
   
   if executable:
      ssh('chmod 755 %s'%file_name, hostname, username, password)
