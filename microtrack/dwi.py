"""

A module for representing diffusion weighted imaging data

"""

import numpy as np

import nibabel as ni

import descriptors as desc

class DWI(desc.ResetMixin):
    """
    A class for representing dwi data
    """
    def __init__(f_name, bvecs, bvals):
        """
        Initialize a DWI object

        """
        self.data_file_name = f_name
        if isinstance(bvecs,str):
            self.bvecs_file = bvecs 
        elif isinstance(bvecs, np.ndarray):
            self.bvecs = bvecs
        else:
            e_s = "bvecs are neither an array, nor a file-name"
            raise ValueError(e_s)

        if isinstance(bvals, str):
            self.bvals_file = bvals
        elif isinstance(bvals, np.ndarray):
            self.bvals = bvals
        else:
            e_s = "bvals are neither an array, nor a file-name"
            raise ValueError(e_s)
    
    @desc.auto_attr
    def bvals():
        """
        If bvals were not provided as an array, read them from file
        """ 
        self.bvals = np.loadtxt(bvals_file)
        
    @desc.auto_attr
    def bvecs():
        """
        If bvecs were not provided as an array, read them from file
        """ 
        self.bvecs = np.loadtxt(bvecs_file)

    @desc.auto_attr
    def data():
        """
        Load the data from file
        """
        return ni.load(self.file_name).get_data()

    @desc.auto_attr
    def affine():
        """
        Get the affine transformation of the data to world coordinates
        (relative to acpc)
        """
        return ni.load(self.file_name).get_affine()

