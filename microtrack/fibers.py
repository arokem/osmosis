# Import from standard lib: 
import struct 
import os

# Import from 3rd party: 
import numpy as np
import nibabel as ni

class Fiber(object):
    """
    This represents a single fiber, its node coordinates and statistics
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
            coordinates and world-based (acpc) coordinates. Defaults to None,
            which implies the identity matrix.

        fiber_stats: dict containing statistics as scalars, corresponding to the
               entire fiber
               
        node_stats: dict containing statistics as np.array, corresponding
            to point-by-point values of the statistic.
            
        """
        coords = np.asarray(coords)
        if len(coords.shape)>2 or coords.shape[0]!=3:
            e_s = "coords input has shape ("
            e_s += ''.join(["%s, "%n for n in coords.shape])
            e_s += "); please reshape to be 3 by n"
            raise ValueError(e_s)

        self.coords = coords

        # Count the nodes
        if len(coords.shape)>1:
            self.n_nodes = coords.shape[0]
        # This is the case in which there is only one coordinate/node:
        else:
            self.n_nodes = 1
            
        if affine is None: 
            self.affine = None # This implies np.eye(4), see below in xform
        elif affine.shape != (4,4):
            # Raise an erro if the affine doesn't make sense:
            e_s = "affine input has shape ("
            e_s += ''.join(["%s, "%n for n in affine.shape])
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

    def xform(self, affine=None, inplace=True):
        """
        Transform the fiber coordinates according to an affine transformation

        Parameters
        ----------
        affine: optional, 4 by 4 matrix
            Per default, the fiber's own affine will be used. If this input is
            provided, this affine will be used instead of the fiber's
            affine, and the new affine will be the inverse of this matrix.

        inplace: optional, bool
            Per default, the transformation occurs inplace, meaning that the
            Fiber is mutated inplace. However, if you don't want that to
            happen, setting this to False will cause this function to return
            another Fiber with transformed coordinates and the inverse of the
            original affine.

        Note
        ----
        Transforming inverts the affine, such that calling xform() twice gives
        you back what you had in the first place.
        
        """
        xyz_orig = self.coords

        # If this is a single point: 
        if len(xyz_orig.shape) == 1:
            xyz_orig1 = np.vstack([np.array([xyz_orig]).T,1])
        else:
            xyz_orig1 = np.vstack([xyz_orig,np.ones(xyz_orig.shape[-1])])
            
        # If the affine optional kwarg was provided use that:
        if affine is None:
            if self.affine is None:
                if inplace:
                    return # Don't do anything and return
                else:
                    # Give me back an identical Fiber:
                    return Fiber(self.coords,
                                 None,
                                 self.fiber_stats,
                                 self.node_stats)
                
            # Use the affine provided on initialization:
            else:
                affine = self.affine
            
        # Do it:
        xyz1 = affine * xyz_orig1

        xyz_new = np.array([np.array(xyz1[0]).squeeze(),
                    np.array(xyz1[1]).squeeze(),
                    np.array(xyz1[2]).squeeze()])

        # Just put the new coords instead of the old ones:
        if inplace:
            self.coords = xyz_new
            # And adjust the affine to be the inverse transform:
            self.affine = affine.getI()
        # Generate a new fiber and return it:
        else: 
            return Fiber(self.coords,
                         affine.getI(),
                         self.fiber_stats,
                         self.node_stats)
            
        
class FiberGroup(object):
    """
    This represents a group of fibers.
    """
    def __init__(self, fibers,
                 name="FG-1",
                 color=[200, 200, 100],
                 thickness=-0.5,
                 affine=None
                 ):
        # XXX Need more stuff (more inputs!) here:
        self.fibers = fibers
        self.n_fibers = len(fibers)
        self.n_nodes = np.sum([f.n_nodes for f in self.fibers])
        # Gather all the unique fiber stat names:
        k_list=[]
        # Get all the keys from each fiber:
        for f in self.fibers:
            k_list += f.fiber_stats.keys()

        # Get a set out (unique values):
        keys = set(k_list)
        # This will eventually hold all the fiber stats:
        self.fiber_stats = {}
        # Loop over unique stat names and... 
        for k in keys:
            # ... put them in a list in each one ...
            self.fiber_stats[k] = []
            # ... from each fiber:
            for f_idx in range(self.n_fibers):
                this_fs = self.fibers[f_idx].fiber_stats
                # But only if that fiber has that stat: 
                if k in this_fs.keys():
                    self.fiber_stats[k].append(this_fs[k])
                # Otherwise, put a nan there:
                else:
                    self.fiber_stats[k].append(np.nan)

        # If you want to give the FG an affine of its own to apply to the
        # fibers in it:
        if affine is not None:
            self.affine = np.matrix(affine)
        else:
            self.affine = None

        self.name = name
    def xform(self, affine=None, inplace=True):
        """
        Transform each fiber in the fiber group according to an affine
        
        Precedence order : input > Fiber.affine > FiberGroup.affine 

        Parameters
        ----------
        affine: 4 by 4 array/matrix
            An affine to apply instead of the affine provided by the Fibers
            themselves and instead of the affine provided by the FiberGroup

        inplace: Whether to change the FiberGroup/Fibers inplace.
        """
        if affine is None:
            affine = self.affine

        if not inplace:
            fibs = np.copy(self.fibers) # Make a copy, to be able to do this not
                                        # inplace
        else:
            fibs = self.fibers
            
        for f in fibs:
            if f.affine is None:
                if affine is not None:
                    # Make sure it's a matrix:
                    affine = np.matrix(affine)
                    # Do it inplace (for the copy, if you made one above): 
                    f.xform(affine)
            else:
                f.xform(f.affine, inplace)

        # If this was self.affine, it will be assigned there now and stick:
        if affine is not None: 
            affine = affine.getI()

        if inplace:
            self.affine = affine

        # If we asked to do things inplace, we are done. Otherwise, we return a
        # FiberGroup
        else:
            return FiberGroup(fibs,
                              name="FG-1",
                              color=[200, 200, 100],
                              thickness=-0.5,
                              affine=affine) # It's already been inverted above
                
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
                   (n_nodes, 3)).T)
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
        fibers.append(Fiber(pts[p_idx],
                            xform,
                            fiber_stats=dict(zip(f_stat_k, f_stat_v)),
                            node_stats=dict(zip(n_stats_k, n_stats_v))))
        
    name = os.path.split(file_name)[-1].split('.')[0]
    return FiberGroup(fibers, name=name)
    
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
