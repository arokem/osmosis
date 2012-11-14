"""

=====================
File input and output
=====================
Reading and writing files from a variety of formats.

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

import numpy as np
import scipy.io as sio

import nibabel as ni
import nibabel.trackvis as tv

import osmosis as oz
import osmosis.fibers as ozf
from .utils import ProgressBar

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
        label_dict[int(this_num.item())] = name[idx].item()

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
