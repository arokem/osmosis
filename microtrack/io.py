# Import from standard lib: 
import struct 
import os

import numpy as np
import microtrack.fibers as mtf

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
    fmt_dict = {'int':['=i', 4],
                'double':['=d', 8],
                'char':['=c', 1],
                'bool':['=?', 1],
                #'uint':['=I', 4],
                }

    fmt_str = fmt_dict[fmt][0]
    fmt_sz = fmt_dict[fmt][1]
    
    out = np.array([struct.unpack(fmt_str,
                    file_read[idx + fmt_sz * j:idx + fmt_sz + fmt_sz * j])[0]
        for j in range(obj_to_read)])
                         
    idx += obj_to_read * fmt_sz
    return out, idx

# XXX The following function is way too long. Break it up!
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
    This only reads Version 3 PDB.
    The file-format is organized as a semi-hierarchical data-base, according to
    the following specification (Also documented here: )
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

    # XXX The algorithms bit is not really working as advertised: 
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
       # XXX The following are not actually encoded in the fiber header and are
         currently set in an arbitrary fashion:
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
    # Read the file as binary info:
    f_read = file(file_name, 'r').read()
    # This is an updatable index into this read:
    idx = 0
    
    # First part is an int encoding the offset (what's that?): 
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
    for stat in range(numstats):
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
    if version != 3:
        raise ValueError("Can only read PDB version 3 files")
    elif verbose:
        print("Loading a PDB version 3 file from: %s"%file_name)

    # How many fibers?
    numpaths, idx = _unpacker(f_read, idx, 1)
    # The next few bytes encode the number of points in each fiber:
    pts_per_fiber, idx = _unpacker(f_read, idx, numpaths)
    total_pts = np.sum(pts_per_fiber)
    # Next we have the xyz coords of the nodes in all fibers: 
    fiber_pts, idx = _unpacker(f_read, idx, total_pts * 3, 'double')

    # We extract the information on a fiber-by-fiber basis
    pts_read = 0 
    pts = []
    for p_idx in range(numpaths):
        n_nodes = pts_per_fiber[p_idx]
        pts.append(np.reshape(
                   fiber_pts[pts_read * 3:(pts_read + n_nodes) * 3],
                   (n_nodes, 3)).T)
        pts_read += n_nodes
        if verbose and np.mod(p_idx, 1000)==0:
            print("Loaded %s of %s paths"%(p_idx, numpaths[0]))            

    f_stats_dict = {}
    for stat_idx in range(numstats):
        per_fiber_stat, idx = _unpacker(f_read, idx, numpaths, 'double')
        f_stats_dict[stats_header["local_name"][stat_idx]] = per_fiber_stat

    n_stats_dict = {}
    for stat_idx in range(numstats):
        pts_read = 0
        if stats_header["computed_per_point"][stat_idx]:
            name = stats_header["local_name"][stat_idx]
            n_stats_dict[name] = []
            per_point_stat, idx = _unpacker(f_read, idx, total_pts, 'double')
            for p_idx in range(numpaths):
                n_stats_dict[name].append(
                    per_point_stat[pts_read : pts_read + pts_per_fiber[p_idx]])
                
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
        fibers.append(mtf.Fiber(pts[p_idx],
                            xform,
                            fiber_stats=dict(zip(f_stat_k, f_stat_v)),
                            node_stats=dict(zip(n_stats_k, n_stats_v))))
    if verbose:
        print("Done reading from file")
        
    name = os.path.split(file_name)[-1].split('.')[0]
    return mtf.FiberGroup(fibers, name=name)
    
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
