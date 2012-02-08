"""
This module implements representations of DWI-derived fibers and fiber groups.
"""

# Import from standard lib: 
import struct 
import os

# Import from 3rd party: 
import numpy as np
import scipy.stats as stats
import scipy.linalg as la

# Import locally: 
import descriptors as desc
import tensor as mtt

class Fiber(desc.ResetMixin):
    """
    This represents a single fiber, its node coordinates and statistics
    """

    def __init__(self, coords, affine=None, fiber_stats=None, node_stats=None):
        """
        Initialize a fiber

        Parameters
        ----------
        coords: np.array of shape 3 x n
            The x,y,z coordinates of the nodes comprising the fiber.

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
        elif affine.shape != (4, 4):
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
            xyz_orig1 = np.vstack([np.array([xyz_orig]).T, 1])
        else:
            xyz_orig1 = np.vstack([xyz_orig, np.ones(xyz_orig.shape[-1])])
            
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

    @desc.auto_attr
    def unique_coords(self):
        """
        What are the unique spatial coordinates in the fiber
        
        Parameters
        ----------
        res: array-like with 3 items for the resolution in the x/y/z
        directions.
        """
        # This is like Matlab's unique(coords, 'rows'), but preserves order(!) 
        return stats._support.unique(self.coords.T).T
    
    def tensors(self, bvecs, bvals, axial_diffusivity, radial_diffusivity):
        """

        The tensors generated by this fiber.

        Parameters
        ----------
        fiber: A Fiber object.

        axial_diffusivity: float
            The estimated axial diffusivity of a single fiber tensor. 

        radial_diffusivity: float
            The estimated radial diffusivity of a single fiber tensor.
         
        Note
        ----
        Estimates of the radial/axial diffusivities may rely on
        empirical measurements (for example, the AD in the Corpus Callosum), or
        may be based on a biophysical model of some kind.
        """

        tensors = []

        grad = np.array(np.gradient(self.coords)[1])
        for this_grad in grad.T:
            u,s,v = la.svd(np.matrix(this_grad.T))
            this_Q = (np.matrix(v) *
                 np.matrix(np.diag([axial_diffusivity,
                                    radial_diffusivity,
                                    radial_diffusivity])) *
                np.matrix(v).T)
            tensors.append(mtt.Tensor(this_Q, bvecs, bvals))
        return tensors

    def predicted_signal(self,
                         axial_diffusivity,
                         radial_diffusivity,
                         S0,
                         bvecs,
                         bvals):
        """
        Compute the fiber contribution to the signal along its coords.


        Notes
        -----
        
        The calculation is based on a simplified Stejskal/Tanner equation:

        .. math::

           S = S0 exp^{-bval (\vec{b}*Q*\vec{b}^T)}

        Where $\vec{b} * Q * \vec{b}^t$ is the ADC for each tensor


        Parameters
        ----------
        """
        tensor_list = self.tensors(bvecs,
                                   bvals,
                                   axial_diffusivity,
                                   radial_diffusivity)
        S = []

        # There's either on S0 per voxel (and possibly bvec)
        if np.iterable(S0) and len(S0) == len(tensor_list):
            for idx, T in enumerate(tensor_list):
                S.append(T.predicted_signal(S0[idx]))

        # Or just one global S0:
        else: 
            for idx, T in enumerate(tensor_list):
                S.append(T.predicted_signal(S0))
                
        return np.array(S) # XXX Consider how to sum over nodes in a single
                            # voxel.


class FiberGroup(object):
    """
    This represents a group of fibers.
    """
    def __init__(self,
                 fibers,
                 name=None,
                 color=None,
                 thickness=None,
                 affine=None
                 ):
        """
        Initialize a group of fibers

        Parameters
        ----------
        fibers: list
            A set of Fiber objects, which will populate this FiberGroup.

        name: str
            Name of this fiber group, defaults to "FG-1"

        color: 3-long array or array like
            RGB for display of fibers from this FiberGroup. Defaults to
            [200, 200, 100]

        thickness: float
            The thickness when displaying this FiberGroup. Defaults to -0.5

        affine: 4 by 4 array or matrix
            Homogenous affine giving relationship between voxel-based
            coordinates and world-based (acpc) coordinates. Defaults to None,
            which implies the identity matrix.
        """
        # Descriptive variables: 

        # Name 
        if name is None:
            name = "FG-1"
        self.name = name

        if color is None:
            color = [200, 200, 100] # RGB
        self.color = np.asarray(color)

        if thickness is None:
            thickness = -0.5
        self.thickness = thickness
        
        # XXX Need more stuff (more inputs!) here:
        self.fibers = fibers
        self.n_fibers = len(fibers)
        self.n_nodes = np.sum([f.n_nodes for f in self.fibers])
        # Gather all the unique fiber stat names:
        k_list = []
        # Get all the keys from each fiber:
        for fiber in self.fibers:
            k_list += fiber.fiber_stats.keys()

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
            in_affine = False  # We need to be able to discriminate between
                               # affines provided by the class instance and
                               # affines provided as inputs
        else:
            in_affine = True

        if not inplace:
            fibs = np.copy(self.fibers) # Make a copy, to be able to do this not
                                        # inplace
        else:
            fibs = self.fibers # Otherwise, save memory by pointing to the same
                               # objects in memory
            
        for this_f in fibs:
            # This one takes the highest precedence: 
            if in_affine:
                this_f.xform(np.matrix(affine))

            # Otherwise, the fiber affines take precedence: 
            elif this_f.affine is not None:
                this_f.xform()
                affine = None # The resulting object should not have an
                              # affine.
                
            # And finally resort to the FG's affine:
            else:
                this_f.xform(self.affine)
                affine = self.affine # Invert the objects affine,
                                        # before assigning to the output

        if affine is not None:
            affine = np.matrix(affine).getI()
            
        if inplace:
            self.fibers = fibs
            self.affine = affine

        # If we asked to do things inplace, we are done. Otherwise, we return a
        # FiberGroup
        else:
            return FiberGroup(fibs,
                              name="FG-1",
                              color=[200, 200, 100],
                              thickness=-0.5,
                              affine=affine) # It's already been inverted above

    def coords(self):
        """
        Hold all the coords from all fibers:
        """
        tmp = []
        for fiber in self.fibers:
            tmp.append(fiber.coords)

        # Concatenate 'em together:
        tmp = np.hstack(tmp)
            
        return tmp
        
    def unique_coords(self):
        """
        The unique spatial coordinates of all the fibers in the FiberGroup
        """
        tmp = []
        for fiber in self.fibers:
            tmp.append(fiber.coords)

        # Concatenate 'em together:
        tmp = np.hstack(tmp)

        tmp = stats._support.unique(tmp.T)
        return np.reshape(tmp,(np.prod(tmp.shape)/3,3)).T

        
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
    for this in arr:
        if this: # The sign that you reached the end of the word is an empty
                 # char
            make_a_word.append(this)
        else:
            break
    return ''.join(make_a_word)
