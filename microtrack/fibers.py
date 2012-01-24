import numpy as np
import nibabel as ni

class Fiber(object):
    """
    This represents a single fiber.

    Should have the following attributes:

    1. X/Y/Z coordinates. 
    2. An affine transformation. 
    3. Empty container for pathway statistics
    4. 
    
    """
    
    def __init__(self, coords, affine=False, stats=False )
        """
        Initialize a fiber

        Parameters
        ----------
        coords: np.array of shape n x 3
            The xyz coordinates of the nodes comprising the fiber.

        affine: np.array of shape 4 x 4
            homogenous affine giving relationship between voxel coordinates and
            world coordinates.

        stats: dict containing statistics as: scalar or np.array, corresponding
            to point-by-point values of the statistic.
            
        """
        self.coords = coords
        self.affine = affine
        self.stats = stats 
        
class FiberGroup(object):
    """
    This represents a group of fibers.
    """
    def __init__(self, fibers):
        raise NotImplementedError
    
def fg_from_pdb(file_name):
    """
    Read the definition of a fiber-group from a .pdb file

    Note
    ----
    This only reads Version 3 PDB.
    
    """
    raise NotImplementedError
