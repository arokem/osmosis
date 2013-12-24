"""

Submissions on an SGE, over ssh

"""

import os
import getpass
import inspect
import traceback

# We need to know whether we have a Qt shell on our hands:
import IPython.zmq.zmqshell as zmqshell

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

class SSH(object):
   """
   A class for representing and using ssh connections via paramiko
   """
   def __init__(self, hostname=None, username=None, password=None, port=22):
      """
      Initialize an ssh connection. Don't connect yet.
      """
      self.hostname = hostname
      self.username = username
      self.password = password
      self.port = port
      self.client = paramiko.SSHClient()
      self.client.load_system_host_keys()
      self.client.set_missing_host_key_policy(paramiko.WarningPolicy)
      # Until the connection is opened, this is set to False:
      self._open = False
      
   def connect(self):
      """
      Open the connection
      """
      # Defer getting the credentials to this point:
      hostname, username, password = self._get_credentials(self.hostname,
                                                           self.username,
                                                           self.password)
      # Update these attrs if need be: 
      self.hostname = hostname
      self.username = username
      self.password = password
      
      self.client.connect(self.hostname, self.port, self.username, self.password)
      self._open = True
      print ("Connected to %s"%hostname)

   def _get_credentials(self, hostname=None, username=None, password=None):
      # Get your hostname and credentials:
      if hostname is None:
         hostname = raw_input('Hostname: ')

      if username is None:
         default_username = getpass.getuser()
         username = raw_input('Username [%s]: ' % default_username)
         if len(username) == 0:
            username = default_username

      # We need to check whether we are in an ipython session, to know how to
      # get the password:
      if password is None: 
         try:
            ip = get_ipython()
            is_ip = True
              
         except NameError: 
            is_ip = False

         # In the Qt terminal, we have to use raw_input:
         if is_ip and isinstance(ip, zmqshell.ZMQInteractiveShell):
            password = raw_input('Password for %s@%s: ' % (username,
                                                              hostname))
         # Otherwise, we can use getpass(preferable, no echo)
         else:
            password = getpass.getpass('Password for %s@%s: ' % (username,
                                                                 hostname))

      return hostname, username, password
    
   def exec_command(self, command):
      """
      Execute a command over ssh.

      If the connection is closed, execute it locally
      """
      if self._open:
         stdin, out, err = self.client.exec_command(command)
         stdout = ''
         for line in out:
            stdout += line

         stderr = ''
         for line in err:
            stderr += line
      else:
         #print command
         status = os.system(command)
         stdin = False
         stdout = False
         # In this case, we just signal whether there was an error in executing
         # the command, without returning stdout/stderr as string:
         if status == 0:
            stderr = False
         else:
            stderr = True 

      return stdin, stdout, stderr


   def disconnect(self):
      try:
         self.client.close()
         self._open = False
         print ("Disconnected from %s"%self.hostname)
      except:
         pass

      

def add_params(s, params_dict):
    """

    Add parameter values from a dict to a code string. This sets the values of
    particular elements of the code, so that many different versions of the
    same code can be run with different parameter settings

    """
    for k,v in params_dict.items():
       # Keep quotes on strings:
       if isinstance(v, str):
          s = '%s="%s"\n'%(k,v) + s
       else: 
          s = "%s = %s\n"%(k,v) + s

    return s


def qsub_cmd(call, name, working_dir='cwd', shell='/bin/bash',
             email=None, mem_usage=35, priority=0,
             flags='', output_dir='sgeoutput'):
   """
   This puts together the qsub command.

   Parameters
   ----------
   call : string
       The call to execute. For python scripts, this will take the form:
       "bashcm.sh %s"%python_script_name, because of bash weirdnesses

   name : string
       A name to assign to the job on the sge and in the output directory.

   working_dir : str
      Where to execute the call. Default: 'cwd' will execute the call directly
      in the home directory

   shell : str
      The path to the shell used to execute the call (can this be '/bin/python?')

   email : str
       Address to send messages to.

   mem_usage  : int
      How much maximal memory to allocate to the job (in GB). Default 4 GB.

   priority : int
      A number between -1024 (lowest priority) and 1023 (highest priority) to
      set the relative priority of different jobs, if there is one. 

   flags : str
      Additional flags to qsub

   output_dir : str
      Where to put the .o and .e files.

   Returns
   -------
   A nicely formatted qsub call packed into a conveniently shaped string.
   
   """
   if email is not None:
      return "qsub -N %s -m a -M %s -o %s -e %s -l h_vmem=%sg -p %d %s -S %s %s"\
      %(name,
      email,
      output_dir,
      output_dir,
      mem_usage,
      priority,
      flags,
      shell,
      call)

   else:
      return "qsub -N %s -m a -o %s -e %s -l h_vmem=%sg -p %d %s -S %s %s"\
      %(name,
      output_dir,
      output_dir,
      mem_usage,
      priority,
      flags,
      shell,
      call)


def _sftp(local_path, remote_path, hostname=None, username=None, password=None,
          port=22, get_or_put='put'):
   """
   Helper function for getting and putting stuff over paramiko sftp
   """
   hostname, username, password = _get_credentials(hostname, username, password)


   transport = paramiko.Transport((hostname, port))
   transport.connect(username = username, password = password)
   sftp = paramiko.SFTPClient.from_transport(transport)

   if get_or_put == 'put' or get_or_put=='up':
      sftp.put(local_path, remote_path)
   elif get_or_put == 'get' or get_or_put=='down':
      sftp.get(remote_path, local_path)
   else:
      raise ValueError("You can either 'get' or 'put', or 'up' or 'down'")

   # Close up:
   sftp.close()
   transport.close()
   
def sftp_up(local_path, remote_path, hostname=None, username=None, password=None,
            port=22):
   """
   Upload a file over ftp
   """
   _sftp(local_path, local_path, hostname=hostname, username=hostname,
         password=password, port=port, get_or_put='put')


def sftp_down(remote_path, local_path, hostname=None, username=None,
              password=None, port=22):
   """
   Download a file over ftp
   """
   _sftp(local_path, remote_path, hostname=hostname, username=hostname,
         password=password, port=port, get_or_put='get')


def py_cmd(ssh, cmd, file_name='~/pycmd/cmd.py', python=None):
   """
   This generates a python script on the cluster side from the string
   contained in cmd + a hash-bang header.
    
   Parameters
   ----------
   ssh : an SSH class instance, with an open connection
   
   cmd : string
       The stuff that goes into the script, can often be multiple lines

   file_name : string
        The name of the file on the cluster.
    
   python : str
       The hash-bang. If none, no hash-bang inserted
   """
   if python is None:
      hdr_line = ''
   else:
      hdr_line = '#!%s'%python
   ssh.exec_command('touch %s'%file_name)
   # The following line over-writes whatever was in that file before: 
   ssh.exec_command("echo '%s' > %s"%(hdr_line, file_name))

   # Make sure quotes get correctly escaped in:
   cmd.replace('"', '\"')
   cmd.replace("'", "\'")
    
   ssh.exec_command("echo '%s' >> %s"%(cmd, file_name))
   ssh.exec_command('chmod 755 %s'%file_name)


def write_file_ssh(ssh, line_list, file_name, executable=True, clobber=True):
   """
   Write a file made out of the lines in line_list on a remote machine

   Note
   ----
   This will clobber existing files! Need to figure out a way to be more
   careful with that.
   """

   if clobber:
      # Does this overwrite existing files? 
      touch_cmd = 'touch %s'%file_name
      i, o, e = ssh.exec_command(touch_cmd)
      if e:
         raise ValueError('error executing "%s"'%touch_cmd)
   
   for line in line_list:
      # From here on out append ('>>')
      echo_cmd = "echo '%s' >> %s"%(line, file_name)
      i,o,e = ssh.exec_command(echo_cmd)
      if e:
         raise ValueError('error executing "%s"'%echo_cmd)

   if executable:
      chmod_cmd = 'chmod 755 %s'%file_name
      i,o,e = ssh.exec_command(chmod_cmd)
      if e:
         raise ValueError('error executing "%s"'%chmod_cmd)
      
      

