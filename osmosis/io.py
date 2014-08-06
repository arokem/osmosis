"""

=====================
File input and output
=====================
Reading and writing files from a variety of formats.


PDB files
---------
The PDB file format is sometimes used for reading and writing information about
tractography results. The *nominal* PDB file format specification is as
follows, but note that some of these things are not implemented in PDB version
3. For example, there are no algorithms to speak of, so that whole bit is
completely ignored. 

 The file-format is organized as a semi-hierarchical data-base, according to
    the following specification:
    [ header size] - int
    -- HEADER FOLLOWS --
    [4x4 xform matrix ] - 16 doubles
    [ number of pathway statistics ] - int
    for each statistic:
        [ currently unused ] - bool
        [ is stat stored per point, or aggregate per path? ] - bool
        [ currently unused ] - bool
        [ name of the statistic ] - char[255]
        [ currently unused ] - char[255]
        [ unique ID - unique identifier for this stat across files ] - int

    ** The algorithms bit is not really working as advertised: **
    [ number of algorithms ] - int
    for each algorithm:
       [ algorithm name ] - char[255]
       [ comments about the algorithm ] - char[255]
       [ unique ID -  unique identifier for this algorithm, across files ] - int

    [ version number ] - int
    -- HEADER ENDS --
    [ number of pathways ] - int
    [ pts per fiber ] - number of pathways integers
    for each pathway:
       [ header size ] - int
       -- PATHWAY HEADER FOLLOWS --
        ** The following are not actually encoded in the fiber header and are
         currently set in an arbitrary fashion: **
       [ number of points ] - int
       [ algorithm ID ] - int
       [ seed point index ] - int

       for each statistic:
          [ precomputed statistical value ] - double
       -- PATHWAY HEADER ENDS --
       for each point:
            [ position of the point ] - 3 doubles (ALREADY TRANSFORMED from
                                                   voxel space!)  
          for each statistic:
             IF computed per point (see statistics header, second bool field):
             for each point:
               [ statistical value for this point ] - double


"""

# Import from standard lib: 
import struct 
import os
import inspect
import warnings
import urllib
import zipfile    

import numpy as np
import scipy.io as sio
import scipy.stats as stats

import nibabel as ni
import nibabel.trackvis as tv


import osmosis as oz
import osmosis.utils as ozu
import osmosis.fibers as ozf
import osmosis.model.dti as dti
from .utils import ProgressBar
import osmosis.volume as ozv

from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table

osmosis_path =  os.path.split(oz.__file__)[0]

data_path = osmosis_path + '/data/'

# XXX The following functions are way too long. Break 'em up!
def fg_from_pdb(file_name, verbose=True):
    """
    Read the definition of a fiber-group from a .pdb file
    Parameters
    ----------
    file_name: str
       Full path to the .pdb file
    Returns
    -------
    A FiberGroup object

    Note
    ----
    This only reads Version 3 PDB. For the full file-format spec, see the
    osmosis.io module top-level docstring
    
    """
    # Read the file as binary info:
    f_obj = file(file_name, 'r')
    f_read = f_obj.read()
    f_obj.close()
    # This is an updatable index into this read:
    idx = 0
    
    # First part is an int encoding the offset to the fiber part: 
    offset, idx = _unpacker(f_read, idx, 1)  

    # Next bit are doubles, encoding the xform (4 by 4 = 16 of them):
    xform, idx  = _unpacker(f_read, idx, 16, 'double')
    xform = np.reshape(xform, (4, 4))
   
    # Next is an int encoding the number of stats: 
    numstats, idx = _unpacker(f_read, idx, 1)

    # The stats header is a dict with lists holding the stat per 
    stats_header = dict(luminance_encoding=[],  # int => bool
                        computed_per_point=[],  # int => bool
                        viewable=[],  # int => bool
                        agg_name=[],  # char array => string
                        local_name=[],  # char array => string
                        uid=[]  # int
        )

    # Read the stats header:
    counter = 0
    while counter < numstats:
        counter += 1
        for k in ["luminance_encoding",
                  "computed_per_point",
                  "viewable"]:
            this, idx = _unpacker(f_read, idx, 1)
            stats_header[k].append(np.bool(this))

        for k in ["agg_name", "local_name"]:
            this, idx = _unpacker(f_read, idx, 255, 'char')
            stats_header[k].append(_word_maker(this))
        # Must have integer reads be word aligned (?): 
        idx += 2
        this, idx = _unpacker(f_read, idx, 1)
        stats_header["uid"].append(this)
    
    # We skip the whole bit with the algorithms and go straight to the version
    # number, which is one int length before the fibers:  
    idx = offset - 4
    version, idx = _unpacker(f_read, idx, 1)
    if int(version) < 2:
        raise ValueError("Can only read PDB version 2 or version 3 files")
    elif verbose:
        print("Loading a PDB version %s file from: %s"%(int(version), file_name))

    if int(version) == 2:
        idx = offset
        
    # How many fibers?
    numpaths, idx = _unpacker(f_read, idx, 1)
    
    if int(version) == 2:
        pts = []
        if verbose:
                prog_bar = ProgressBar(numpaths[0])
                f_name = inspect.stack()[0][3]

        f_stats = []
        n_stats = []
        for p_idx in range(numpaths):
            f_stats_dict = {}
            n_stats_dict = {}

            # Keep track of where you are right now
            ppos = idx
            path_offset, idx = _unpacker(f_read, idx, 1)
            n_nodes, idx = _unpacker(f_read, idx, 1)
            # As far as I can tell the following two don't matter much:
            algo_type, idx = _unpacker(f_read, idx, 1)
            seed_pt_idx, idx = _unpacker(f_read, idx, 1)
            # Read out the per-path stats:
            for stat_idx in range(numstats):
                per_fiber_stat, idx = _unpacker(f_read, idx, 1, 'double')
                f_stats_dict[stats_header["local_name"][stat_idx]] = \
                    per_fiber_stat
            f_stats.append(f_stats_dict)
            # Skip forward to where the paths themselves are:
            idx = ppos
            # Read the nodes:
            pathways, idx = _unpacker(f_read, idx, n_nodes*3, 'double')
            
            pts.append(np.reshape(pathways, (n_nodes, 3)).T)
            for stat_idx in range(numstats):
                if stats_header["computed_per_point"][stat_idx]:
                    name = stats_header["local_name"][stat_idx]
                    n_stats_dict[name], idx = _unpacker(f_read, idx, n_nodes,
                                                    'double')

            n_stats.append(n_stats_dict)
            
        fibers = []
            
        # Initialize all the fibers:
        for p_idx in range(numpaths):
            this_fstats_dict = f_stats[p_idx]
            f_stat_k = this_fstats_dict.keys()
            f_stat_v = [this_fstats_dict[k] for k in f_stat_k]
            this_nstats_dict = n_stats[p_idx]
            n_stats_k = this_nstats_dict.keys()
            n_stats_v = [this_nstats_dict[k] for k in n_stats_k]
            fibers.append(ozf.Fiber(pts[p_idx],
                                xform,
                                fiber_stats=dict(zip(f_stat_k, f_stat_v)),
                                node_stats=dict(zip(n_stats_k, n_stats_v))))
            
            
    elif int(version) == 3: 
        # The next few bytes encode the number of points in each fiber:
        pts_per_fiber, idx = _unpacker(f_read, idx, numpaths)
        total_pts = np.sum(pts_per_fiber)
        # Next we have the xyz coords of the nodes in all fibers: 
        fiber_pts, idx = _unpacker(f_read, idx, total_pts * 3, 'double')

        # We extract the information on a fiber-by-fiber basis
        pts_read = 0 
        pts = []

        if verbose:
            prog_bar = ProgressBar(numpaths[0])
            f_name = inspect.stack()[0][3]
        for p_idx in range(numpaths):
            n_nodes = pts_per_fiber[p_idx]
            pts.append(np.reshape(
                       fiber_pts[pts_read * 3:(pts_read + n_nodes) * 3],
                       (n_nodes, 3)).T)
            pts_read += n_nodes
            if verbose:
                prog_bar.animate(p_idx, f_name=f_name)            

        f_stats_dict = {}
        for stat_idx in range(numstats):
            per_fiber_stat, idx = _unpacker(f_read, idx, numpaths, 'double')
            # This is a fiber-stat only if it's not computed per point:
            if not stats_header["computed_per_point"][stat_idx]: 
                f_stats_dict[stats_header["local_name"][stat_idx]] =\
                    per_fiber_stat

        per_point_stat = []
        n_stats_dict = {}
        for stat_idx in range(numstats):
            pts_read = 0
            # If it is computer per point, it's a node-stat:
            if stats_header["computed_per_point"][stat_idx]:
                name = stats_header["local_name"][stat_idx]
                n_stats_dict[name] = []
                per_point_stat, idx = _unpacker(f_read, idx, total_pts, 'double')
                for p_idx in range(numpaths):
                    n_stats_dict[name].append(
                        per_point_stat[pts_read:pts_read + pts_per_fiber[p_idx]])

                    pts_read += pts_per_fiber[p_idx]
            else:
                per_point_stat.append([])

        fibers = []
        # Initialize all the fibers:
        for p_idx in range(numpaths):
            f_stat_k = f_stats_dict.keys()
            f_stat_v = [f_stats_dict[k][p_idx] for k in f_stat_k]
            n_stats_k = n_stats_dict.keys()
            n_stats_v = [n_stats_dict[k][p_idx] for k in n_stats_k]
            fibers.append(ozf.Fiber(pts[p_idx],
                                xform,
                                fiber_stats=dict(zip(f_stat_k, f_stat_v)),
                                node_stats=dict(zip(n_stats_k, n_stats_v))))
    if verbose:
        print("Done reading from file")
        
    name = os.path.split(file_name)[-1].split('.')[0]
    return ozf.FiberGroup(fibers, name=name, affine=xform)
    
# This one's a global used in both packing and unpacking the data 
    
_fmt_dict = {'int':['=i', 4],
             'double':['=d', 8],
             'char':['=c', 1],
             'bool':['=?', 1],
             #'uint':['=I', 4],
                }

def pdb_from_fg(fg, file_name='fibers.pdb', verbose=True, affine=None):
    """
    Create a pdb file from a osmosis.fibers.FiberGroup class instance.

    Parameters
    ----------
    fg: a FiberGroup object

    file_name: str
       Full path to the pdb file to be saved.
    
    """

    fwrite = file(file_name, 'w')

    # The total number of stats are both node-stats and fiber-stats:
    n_stats = len(fg[0].fiber_stats.keys()) + len(fg[0].node_stats.keys())
    stats_hdr_sz = (4 * _fmt_dict['int'][1] + 2 * _fmt_dict['char'][1] * 255 + 2)

    
    # This is the 'offset' to the beginning of the fiber-data. Note that we are
    # just skipping the whole algorithms thing, since that seems to be unused
    # in mrDiffusion anyway. 
    hdr_sz = (4 * _fmt_dict['int'][1] + # ints: hdr_sz itself, n_stats, n_algs
                                        # (always 0), version
             16 *_fmt_dict['double'][1] +      # doubles: the 4 by 4 affine
             n_stats * stats_hdr_sz) # The stats part of the header, add
                                        # one for good measure(?).

    
    _packer(fwrite, hdr_sz)
    if affine is None:
        if fg.affine is None:
            affine = tuple(np.eye(4).ravel().squeeze())
        else:
            affine = tuple(np.array(fg.affine).ravel().squeeze())
    else:
        affine = tuple(np.array(affine).ravel().squeeze())
        
    _packer(fwrite, affine, 'double')
    _packer(fwrite, n_stats)

    
    # We are going to assume that fibers are homogenous on the following
    # properties. XXX Should we check that when making FiberGroup instances? 
    uid = 0
    for f_stat in fg[0].fiber_stats:
        _packer(fwrite, True)   # currently unused
        _packer(fwrite, False)  # Is this per-point?
        _packer(fwrite, True)   # currently unused
        _stat_hdr_set(fwrite, f_stat, uid)
        uid += 1  # We keep tracking that across fiber and node stats
        
    for n_stat in fg[0].node_stats:
        # Three True bools for this one:
        for x in range(3):
            _packer(fwrite, True)
        _stat_hdr_set(fwrite, n_stat, uid)
        uid += 1

    _packer(fwrite, 0) # Number of algorithms - set to 0 always
     
    fwrite.seek(hdr_sz - _fmt_dict['int'][1])
    
    # This is the PDB file version:
    _packer(fwrite, 3)
    _packer(fwrite, fg.n_fibers)
    
    for fib in fg.fibers:
        # How many coords in each fiber:
        _packer(fwrite, fib.coords.shape[-1])

    # x,y,z coords in each fiber:
    for fib in fg.fibers:
        _packer(fwrite, fib.coords.T.ravel(), 'double')

    for stat in fg[0].fiber_stats:
        for fib in fg.fibers:
            _packer(fwrite, fib.fiber_stats[stat],'double')

    # The per-node stats have to be inserted in here as well, with their mean
    # value: 
    for stat in fg[1].node_stats:
        for fib in fg.fibers:
            _packer(fwrite, np.mean(fib.node_stats[stat]), 'double')
     
    for stat in fg[1].node_stats:
        for fib in fg.fibers:
            _packer(fwrite, fib.node_stats[stat], 'double')
    
    if verbose:
        "Done saving data in file%s"%file_name

    fwrite.close()

def _unpacker(file_read, idx, obj_to_read, fmt='int'):

    """
    Helper function to unpack binary data from files with the struct library.

    Relies on http://docs.python.org/library/struct.html

    Parameters
    ----------
    file_read: The output of file.read() from a file object
    idx: An index into x
    obj_to_read: How many objects to read
    fmt: A format string, telling us what to read from there 
    """
    # For each one, this is [fmt_string, size] 
    
    fmt_str = _fmt_dict[fmt][0]
    fmt_sz = _fmt_dict[fmt][1]
    
    out = np.array([struct.unpack(fmt_str,
                    file_read[idx + fmt_sz * j:idx + fmt_sz + fmt_sz * j])[0]
        for j in range(obj_to_read)])
                         
    idx += obj_to_read * fmt_sz
    return out, idx

def _packer(file_write, vals, fmt='int'):
    """
    Helper function to pack binary data to files, using the struct library:

    Relies on http://docs.python.org/library/struct.html    
    
    """
    fmt_str = _fmt_dict[fmt][0]
    if np.iterable(vals):
        for pack_this in vals:
            s = struct.pack(fmt_str, pack_this)
            file_write.write(s)

    else:
        s = struct.pack(fmt_str, vals)
        file_write.write(s)
    
def _word_maker(arr):
    """
    Helper function Make a string out of pdb stats header "name" variables 
    """
    make_a_word = []
    for this in arr:
        if this: # The sign that you reached the end of the word is an empty
                 # char
            make_a_word.append(this)
        else:
            break
    return ''.join(make_a_word)
    
def _char_list_maker(name):
    """
    Helper function that does essentially the opposite of _word_maker. Takes a
    string and makes it into a 255 long list of characters with the name of a
    stat, followed by a single white-space and then 'g' for the rest of the 255 
    """

    l = list(name)
    l.append('\x00')  # The null character
    while len(l)<255:
        l.append('g')
    return l

def _stat_hdr_set(fwrite, stat, uid):
    """
    Helper function for writing stuff into stats header portion of pdb files
    """
    # Name of the stat: 
    char_list = _char_list_maker(stat)
    _packer(fwrite, char_list, 'char')  
    _packer(fwrite, char_list, 'char')  # Twice for some reason
    _packer(fwrite, ['g','g'], 'char')  # Add this, so that that the uid ends
                                        # up "word-aligned".

    _packer(fwrite, uid) # These might get reordered upon
                         # resaving on different platforms, because
                         # dict keys come in no particular order...


def fg_from_trk(trk_file, affine=None):
    """
    Read data from a trackvis .trk file and create a FiberGroup object
    according to the information in it.
    """

    # Generate right away, since we're going to do it anyway:
    read_trk = tv.read(trk_file, as_generator=False)
    fibers_trk = read_trk[0]

    # Per default read from the affine from the file header:
    if affine is not None:
        aff = affine
    else: 
        hdr = read_trk[1]
        aff= tv.aff_from_hdr(hdr)
        # If the header contains a bogus affine, we revert to np.eye(4), so we
        # don't get into trouble later:
        try:
            np.matrix(aff).getI()
        except np.linalg.LinAlgError:
            e_s = "trk file contains bogus header, reverting to np.eye(4)" 
            warnings.warn(e_s)
            aff = np.eye(4)

    fibers = []
    for f in fibers_trk:
        fibers.append(ozf.Fiber(np.array(f[0]).T,affine=aff))

    return ozf.FiberGroup(fibers, affine=aff)

def trk_from_fg(fg, trk_file, affine=None):
    """
    Save a trk file from a FiberGroup class instance

    Note
    ----
    Stats?
    
    """

    # XXX Something to consider when implementing this: We might want to stick
    # the stats into the header somehow. That would also imply making changes
    # to fg_from_trk to allow reading out these stats, if they exist (if osmosis
    # created this file...).  

    raise NotImplementedError


def freesurfer_labels():
    """
    Get the freesurfer labels for different parts of the brain from a file
    stored with the data.

    Parameters
    ----------
    None. 

    Returns
    -------
    A dict with label-name key-value pairs.
    
    """
    data_path = oz.__path__[0] + '/data/'
    fslabel = sio.loadmat(data_path + 'fslabel.mat', squeeze_me=True)['fslabel']
    num = fslabel['num'].item()
    name = fslabel['name'].item()

    label_dict = {}
    for idx, this_num in enumerate(num):
        label_dict[int(this_num)] = str(name[idx])

    return label_dict


def nii_from_volume(vol, file_name, affine=None):
    """
    Create a nifti file from some volume

    Parameters
    ----------
    vol: ndarray
       The data to put in the file

    file_name: str
        Full path

    affine: 4 by 4 array/matrix
       The affine transformation to world/acpc coordinates. Default to np.eye(4)
    

    Returns
    -------
    """

    if affine is None:
        affine = np.eye(4)

    ni.Nifti1Image(vol,affine).to_filename(file_name)



def make_data_set(root, subject):
    """
    Create the full paths to the data given a root
    """ 
    exts = ['.nii.gz', '.bvecs', '.bvals']
    dwi, bvecs, bvals = [os.path.join(data_path, subject, root + ext)
                         for ext in exts]
    return dwi, bvecs, bvals 

def get_dwi_data(b, subject):
    """
    A function that gets you file-names to a data-set with a certain b value
    provided as input
    """
    return [make_data_set(subject + '_b%s_1'%b , subject),
            make_data_set(subject + '_b%s_2'%b, subject)]
    

def download_data():
    """
    This function downloads the data from arokem.org

    Needs to be run once before running any of the analyses that require the
    main DWI data set
    
    """
    print ("Downloading the data from arokem.org...")
    f=urllib.urlretrieve("http://arokem.org/data/osmosis_data.zip")[0]
    zf = zipfile.ZipFile(f)
    zf.extractall(path=osmosis_path)


def get_t1(subject, resample=None):
    """
    Get the high-res T1-weighted anatomical scan. If requested, resample it to
    the resolution of a nifti file for which the name is provided as input
    """
    t1=data_path + '/%s/%s_t1.nii.gz'%(subject, subject)
    t1_nii = ni.load(t1)
    if resample is not None:
        return ozv.resample_volume(t1, resample).get_data()
    else:
        return ni.load(t1).get_data()


def get_brain_mask(bm=data_path + 'brainMask.nii.gz',resample=None):
    """
    Get the brain mask (derived from DWI using mrDiffusion) and resample if
    needed
    """
    bm_nii = ni.load(bm)
    if resample is not None:
        return ozv.resample_volume(bm, resample).get_data()
    else:
        return ni.load(bm).get_data()


def get_wm_mask(wm=data_path + 'SUB1_wm_mask.nii.gz', resample=None):
    """
    Get me a white matter mask. Resample if need be
    """
    bm_nii = ni.load(wm)
    if resample is not None:
        return ozv.resample_volume(wm, resample).get_data()
    else:
        return ni.load(wm).get_data()


def get_ad_rd(subject, b):
    """
    This is a short-cut to get the axial and radial diffusivity values that we
    have extracted from the data with the notebook GetADandRD.
    """
    if subject == 'SUB1':
        diffusivities = {1000:[dict(AD=1.6769,  RD=0.3196),
                           dict(AD=1.6643, RD=0.3177)],
                     2000:[dict(AD=1.3798, RD=0.2561),
                           dict(AD=1.3802, RD=0.2580)],
                     4000:[dict(AD=0.8494, RD=0.2066),
                           dict(AD=0.8478, RD=0.2046)]}

    elif subject == 'SUB2':
        diffusivities = {1000:[dict(AD=1.7646,  RD=0.4296),
                           dict(AD=1.7509, RD=0.4299)],
                     2000:[dict(AD=1.4670, RD=0.5156),
                           dict(AD=1.4710, RD=0.5163)],
                     4000:[dict(AD=0.8288, RD=0.2970),
                           dict(AD=0.8234, RD=0.3132)]}

    return diffusivities[b]


def make_wm_mask(seg_path, dwi_path, out_path=data_path + 'wm_mask.nii.gz',
                 n_IQR=2):
    """

    Function that makes a white-matter mask from an itk-gray class file and DWI
    data, at the resolution of the DWI data.

    Parameters
    ----------
    seg_path : str
       Full path to an itk-gray segmentation

       
    dwi_path : str
        Full path to some DWI data (used to determine the mean diffusivity in
        each voxel, see below)

    out_path : str (optional)
        Where to put the resulting file
        
    n_IQR: float (optional)
        How many IQRs away for the definition of MD outliers (see below) 
    

    Note
    ----

    This follows [Jeurissen2012]_. First, we resample a segmentation based on a
    T1-weighted image. Then, we calculate the mean diffusivity in a
    coregistered DWI image. We then also exclude from the segmentation white
    matter voxels that have

         MD > median (MD) + n_IQR * IQR (MD)

    Where IQR is the interquartile range. Note that per default we take a
    slightly more restrictive criterion (of 2 * IQR, instead of 1.5 * IQR),
    based on some empirical looking at data. Not so sure why there would be a
    difference.

    [Jeurissen2012]Jeurissen B, Leemans A, Tournier, J-D, Jones, DK and Sijbers
    J (2012). Investigating the prevalence of complex fiber configurations in
    white matter tissue with diffusion magnetic resonance imaging. Human Brain
    Mapping. doi: 10.1002/hbm.2209

    """
    dwi_nii, dwi_bvecs, dwi_bvals = make_data_set(dwi_path) 
    # Resample the segmentation image to the DWI resolution:
    seg_resamp = ozv.resample_volume(seg_path, dwi_nii)
    # Find WM voxels (this is ITK gray coding 3=LHWM, 4=RHWM)
    wm_idx = np.where(np.logical_or(seg_resamp==3, seg_resamp==4))
    vol = np.zeros(seg_resamp.shape)
    vol[wm_idx] = 1

    # OK - now we need to find and exclude MD outliers: 
    TM = ozm.TensorModel(dwi_nii, dwi_bvecs, dwi_bvals, mask=vol,
                         params_file='temp')    

    MD = TM.mean_diffusivity
    
    IQR = (stats.scoreatpercentile(MD[np.isfinite(MD)],75) -
          stats.scoreatpercentile(MD[np.isfinite(MD)],25))
    cutoff = np.median(MD[np.isfinite(MD)]) + n_IQR * IQR
    cutoff_idx = np.where(MD > cutoff)

    # Null 'em out
    vol[cutoff_idx] = 0

    # Now, let's save some output:
    ni.Nifti1Image(vol, dwi_ni.get_affine()).to_filename(out_path)

def place_files(file_names, mask_vox_num, expected_file_num, mask_data,
                data, bvals, file_path=os.getcwd(), vol=False,
                f_type="npy", save=False):
    """
    Function to aggregate sub data files from parallelizing.  Assumes that
    the sub_files are in the format: (file name)_(number of sub_file).(file_type)
    
    Parameters
    ----------
    file_names: list
        List of strings indicating the base file names for each output data
        aggregation
    mask_vox_num: int
        Number of voxels in each sub file
    expected_file_num: int
        Expected number of sub files
    mask_data_file: obj
        File handle for brain/white matter mask
    file_path: str
        Path to the directory with all the sub files.  Default is the current
        directory
    vol: str
        String indicating whether or not the sub files are in volumes and
        whether the output files are saved as volumes as well
    f_type: str
        String indicating the type of file the sub files are saved as
    save: str
        String indicating whether or not to save the output aggregation/volumes
    num_dirs: int
        Number of directions in each output aggregation/volume
    
    Returns
    -------
    missing_files: 1 dimensional array
        All the sub files that are missing in the aggregation
    aggre_list: list
        List with all the aggregations/volumes
    """
    files = os.listdir(file_path)

    # Get data and indices
    mask_idx = np.where(mask_data)
    
    bval_list, b_inds, unique_b, bvals_scaled = ozu.separate_bvals(bvals)
    all_b_idx = np.where(bvals_scaled != 0)

    S0 = np.mean(data[..., b_inds[0]],-1)
    pre_mask = np.array(mask_data, dtype=bool)
    ravel_mask = np.ravel(pre_mask).astype(int)
    ravel_mask[np.where(ravel_mask)[0][np.where(S0[pre_mask] == 0)]] = 2
    ravel_mask = ravel_mask[np.where(ravel_mask != 0)]
    ravel_mask[np.where(ravel_mask == 2)] = 0
    ravel_mask = ravel_mask.astype(bool)
    
    aggre_list = []
    missing_files_list = []
    for fn in file_names:
        count = 0
        # Keep track of files in case there are any missing ones
        i_track = np.ones(expected_file_num)
        
        # If you don't want to put the voxels back into a volume, just preallocate
        # enough for each voxel included in the mask.
           
        for f_idx in np.arange(len(files)):
            this_file = files[f_idx]
            if this_file[(len(this_file)-len(f_type)):len(this_file)] == f_type:
                
                if f_type == "npy":
                    sub_data = np.load(os.path.join(file_path, this_file))
                            
                elif f_type == "nii.gz":
                    sub_data = ni.load(os.path.join(file_path, this_file)).get_data()
 
                # If the name of this file is equal to file name that you want to
                # aggregate, load it and find the voxels corresponding to its location
                # in the given mask.
                if this_file[0:len(fn)] == fn:
                    if count == 0:
                        if len(sub_data.shape) == 1:
                            num_dirs = 1

                        else:
                            num_dirs = sub_data.shape[-1]
                            
                        if vol is False:
                            aggre = np.squeeze(ozu.nans((int(np.sum(mask_data)),) + (num_dirs,)))
                        else:
                            aggre = np.squeeze(ozu.nans((mask_data_file.shape + (num_dirs,))))
                    
                    count = count + 1                            
                    i = int(this_file.split(".")[0][len(fn):])
                    low = i*mask_vox_num
                    high = np.min([(i+1) * mask_vox_num,
                                int(np.sum(mask_data))])
                        
                    if vol is False:
			if sub_data.shape[0] > aggre[low:high][ravel_mask[low:high]].shape[0]:
			    aggre[low:high][ravel_mask[low:high]] = np.squeeze(sub_data)[ravel_mask[low:high]]
                        else:
                            aggre[low:high][ravel_mask[low:high]] = np.squeeze(sub_data)
                    else:
                        mask = np.zeros(mask_data_file.shape)
                        mask[mask_idx[0][low:high],
                             mask_idx[1][low:high],
                             mask_idx[2][low:high]] = 1
                        aggre[np.where(mask)] = sub_data
                    # If the file is present, change its index within the tracking array to 0.    
                    i_track[i] = 0
        
        missing_files_list.append(np.squeeze(np.where(i_track)))
        aggre_list.append(aggre)
        
        if save is True:
            if vol is False:
                np.save("aggre_%s.npy"%fn, aggre)
            else:
                aff = mask_data_file.get_affine()
                ni.Nifti1Image(aggre, aff).to_filename("vol_%s.nii.gz"%fn)            
    
    return missing_files_list, aggre_list

def rm_ventricles(wm_data_file, bvals, bvecs, data, data_path):
    """
    Removes the ventricles from the white matter mask
    """
    # Separate b-values and find the indices corresponding to the b0 and
    # bNk measurements where N = the lowest b-value other than 0
    wm_data = wm_data_file.get_data()
    bval_list, b_inds, unique_b, bvals_scaled = ozu.separate_bvals(bvals)
    all_b_idx = np.where(bvals_scaled != 0)
    bNk_b0_inds = np.concatenate((b_inds[0], b_inds[1]))
    
    # Fit a tensor model 
    tm = dti.TensorModel(data[..., bNk_b0_inds], bvecs[:, bNk_b0_inds],
                         bvals[bNk_b0_inds], mask=wm_data,
                         params_file = "temp")
                         
    # Find the median, and 25th and 75th percentiles of mean diffusivities
    md_median = np.median(tm.mean_diffusivity[np.where(wm_data)])
    q1 = stats.scoreatpercentile(tm.mean_diffusivity[np.where(wm_data)],25)
    q3 = stats.scoreatpercentile(tm.mean_diffusivity[np.where(wm_data)],75)
    
    # Exclude voxels with MDs above median + 2*interquartile range
    md_exclude = md_median + 2*(q3 - q1)
    md_include = np.where(tm.mean_diffusivity[np.where(wm_data)] < md_exclude)
    new_wm_mask = np.zeros(wm_data.shape)
    new_wm_mask[np.where(wm_data)[0][md_include],
                np.where(wm_data)[1][md_include],
                np.where(wm_data)[2][md_include]] = 1

    wm = ni.Nifti1Image(new_wm_mask, wm_data_file.get_affine())
    ni.save(wm, os.path.join(data_path, 'wm_mask_no_vent.nii.gz'))
    return new_wm_mask
