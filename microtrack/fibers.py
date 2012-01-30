import numpy as np
import nibabel as ni
import struct 

class Fiber(object):
    """
    This represents a single fiber.

    Should have the following attributes:

    1. X/Y/Z coordinates. 
    2. An affine transformation. 
    3. Empty container for pathway statistics
    4. 
    
    """
    
    def __init__(self, coords, affine=None, fiber_stats=None, node_stats=None):
        """
        Initialize a fiber

        Parameters
        ----------
        coords: np.array of shape 3 x n
            The xyz coordinates of the nodes comprising the fiber.

        affine: np.array of shape 4 x 4
            homogenous affine giving relationship between voxel-based
            coordinates and world-based (acpc) coordinates. 

        fiber_stats: dict containing statistics as scalars, corresponding to the
               entire fiber
               
        node_stats: dict containing statistics as np.array, corresponding
            to point-by-point values of the statistic.
            
        """
        if len(coords.shape)>2 or coords.shape[-1]!=3:
            e_s = "coords input has shape ("
            e_s += ''.join(["%s, "%n for n in coords.shape])
            e_s += "); please reshape to be n by 3"
            raise ValueError(e_s)

        self.coords = coords

        # Count the nodes
        if len(coords.shape)>1:
            self.nnodes = coords.shape[0]
        # This is the case in which there is only one coordinate/node:
        else:
            self.nnodes = 1
            
        if affine is None: 
            # Set to default value: the identity matrix
            affine = np.matrix(np.eye(4))
        elif affine.shape != (4,4):
            # Raise an erro if the affine doesn't make sense:
            e_s = "affine input has shape ("
            e_s += str(["%s, "%n for n in affine.shape])
            e_s += "); please reshape to be 4 by 4"
            raise ValueError(e_s)
        else:
            self.affine = np.matrix(affine)

        if fiber_stats is not None:
            self.fiber_stats = fiber_stats
        else:
            # The default
            self.fiber_stats = {}

        if node_stats is not None:
            self.node_stats = node_stats
        else:
            # The default
            self.node_stats = {}
    
    def xform(self, affine=None, inplace=False):
        """
        Transform the fiber coordinates to 
        """
        
        raise NotImplementedError
    
class FiberGroup(object):
    """
    This represents a group of fibers.
    """
    def __init__(self, fibers):
        # XXX Need more stuff (more inputs!) here:
        self.fibers = fibers

def _unpacker(x, i, n, fmt='int'):

    """
    Helper function to unpack binary data from files with the struct library.

    Relies on http://docs.python.org/library/struct.html

    Parameters
    ----------
    x: The output of file.read() from a file object
    i: An index into x
    n: How many objects to read
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
                                  x[i + fmt_sz * j:i + fmt_sz + fmt_sz * j])[0]
        for j in range(n)])
                         
    i = i + n * fmt_sz
    return out, i

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
    f = file(file_name, 'r')
    f_read = f.read()
    # This is an updatable index into this read:
    i = 0
    
    # First part is an int encoding the offset (what's that?): 
    offset, i = _unpacker(f_read, i, 1)  

    # Next bit are doubles, encoding the xform (4 by 4 = 16 of them):
    xform, i  = _unpacker(f_read, i, 16, 'double')
    xform = np.reshape(xform, (4,4))
   
    # Next is an int encoding the number of stats: 
    numstats, i = _unpacker(f_read, i, 1)

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
            this, i = _unpacker(f_read, i, 1)
            stats_header[k].append(np.bool(this))

        for k in ["agg_name", "local_name"]:
            this, i = _unpacker(f_read, i, 255, 'char')
            stats_header[k].append(_word_maker(this))
        # Must have integer reads be word aligned (?): 
        i += 2
        this, i = _unpacker(f_read, i, 1)
        stats_header["uid"].append(this)

    # We skip the whole bit with the algorithms and go straight to the version
    # number, which is one int length before the fibers:  
    i = offset - 4
    version, i = _unpacker(f_read, i, 1)
    if version != 3:
        raise ValueError("Can only read PDB version 3 files")
    elif verbose:
        print("Loading a PDB version 3 file from: %s"%file_name)

    # How many fibers?
    numpaths, i = _unpacker(f_read, i, 1)
    # The next few bytes encode the number of points in each fiber:
    pts_per_fiber, i = _unpacker(f_read, i, numpaths)
    total_pts = np.sum(pts_per_fiber)
    # Next we have the xyz coords of the nodes in all fibers: 
    fiber_pts, i = _unpacker(f_read, i, total_pts * 3, 'double')

    # We extract the information on a fiber-by-fiber basis
    pts_read = 0 
    pts = []
    for p_idx in range(numpaths):
        n_nodes = pts_per_fiber[p_idx]
        pts.append(np.reshape(
                   fiber_pts[pts_read * 3:(pts_read + n_nodes) * 3],
                   (3, n_nodes)))
        pts_read += n_nodes
        if verbose and np.mod(p_idx+1, 1000)==0:
            print("Loaded %s of %s paths"%(p_idx, numpaths[0]))            

    f_stats_dict = {}
    for stat_idx in range(numstats):
        per_fiber_stat, i = _unpacker(f_read, i, numpaths, 'double')
        f_stats_dict[stats_header["local_name"][stat_idx]] = per_fiber_stat

    n_stats_dict = {}
    for stat_idx in range(numstats):
        pts_read = 0
        if stats_header["computed_per_point"][stat_idx]:
            name = stats_header["local_name"][stat_idx]
            n_stats_dict[name] = []
            per_point_stat, i = _unpacker(f_read, i, total_pts, 'double')
            for p_idx in range(numpaths):
                n_stats_dict[name].append(
                    per_point_stat[pts_read : pts_read + pts_per_fiber[p_idx]])
                
                pts_read += pts_per_fiber[p_idx]
        else:
            fiber_pts_stat.append([])

    fibers = []
    # Initialize all the fibers:
    for p_idx in range(numpaths):
        f_stat_k = f_stats_dict.keys()
        f_stat_v = [f_stats_dict[k][p_idx] for k in f_stat_k]
        n_stats_k = n_stats_dict.keys()
        n_stats_v = [n_stats_dict[k][p_idx] for k in n_stats_k]
        fibers.append(Fiber(pts[p_idx].T,
                            xform,
                            fiber_stats=dict(zip(f_stat_k, f_stat_v)),
                            node_stats=dict(zip(n_stats_k, n_stats_v))))
                
    return FiberGroup(fibers)
    
def _word_maker(arr):
    """
    Helper function Make a string out of pdb stats header "name" variables 
    """
    make_a_word = [] 
    for x in arr:
        if x: # The sign that you reached the end of the word is an empty char 
            make_a_word.append(x)
        else:
            break
    return ''.join(make_a_word)
