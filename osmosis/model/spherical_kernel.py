
from osmosis.model.base import BaseModel, SCALE_FACTOR


class SphericalKernelModel(BaseModel):

    """

    This model predicts the signal in any point on the sphere by averaging the
    nearest neighbors to that point. Averaging the signal is based on a
    spherical analogue of a Gaussian function and cross-validation is used in
    every voxel to estimate the value of this single model parameter. 

    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 params_file=None,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR):

        
        # Initialize the super-class:
        BaseModel.__init__(self,
                           data,
                           bvecs,
                           bvals,
                           affine=affine,
                           mask=mask,
                           scaling_factor=scaling_factor,
                           params_file=params_file)



    def model_params(self):
        """
        The model params here are a single number per voxel, referring to the
        width of the Gaussian that you would need to employ to predict the
        signal. 
        """
        pass
        

    def fit(self):
        """
        
        """
        pass
