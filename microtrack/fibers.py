"""
This module implements representations of DWI-derived fibers and fiber groups.
"""
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
        tens = self.tensors(bvecs,
                            bvals,
                            axial_diffusivity,
                            radial_diffusivity)
        
        ADC = np.array([ten.ADC for ten in tens])

        # Call S/T with the ADC as input:
        return mtt.stejskal_tanner(S0, bvecs, bvals, ADC=ADC)

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

    def __getitem__(self, i):
        """
        Overload __getitem__ to return the i'th fiber when indexing.
        """
        return self.fibers[i]
    
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
        The unique spatial coordinates of all the fibers in the FiberGroup.

        XXX This is hella slow.
        """
        tmp = []
        for fiber in self.fibers:
            tmp.append(fiber.coords)

        # Concatenate 'em together:
        tmp = np.hstack(tmp)

        tmp = stats._support.unique(tmp.T)
        return np.reshape(tmp,(np.prod(tmp.shape)/3,3)).T

