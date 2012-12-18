import numpy as np

import nibabel as ni

import osmosis.utils as ozu
import osmosis.descriptors as desc
from osmosis.model.canonical_tensor import CanonicalTensorModel, AD, RD
from osmosis.model.base import SCALE_FACTOR
from osmosis.model.io import params_file_resolver
                       

class TissueFractionModel(CanonicalTensorModel):
    """
    This is an extension of the CanonicalTensorModel, based on Mezer et al.'s
    measurement of the tissue fraction in different parts of the brain
    [REF?]. The model posits that tissue fraction accounts for non-free water,
    restriced or hindered by tissue components, which can be represented by a
    canonical tensor and a sphere. The rest (1-tf) is free water, which is
    represented by a second sphere (free water).

    Thus, the model is as follows: 

    .. math:

    \begin{pmatrix} D_1 \\ D_2 \\ ... \\D_n \\ TF \end{pmatrix} =

    \begin{pmatrix} T_1 & D_g & D_iso \\ T_2 & D_g & D_iso \\ T_n & D_g & D_iso
    \\ ... & ... & ... \\ \lambda_1 & \lambda_2 & 0 \end{pmatrix}
    \begin{pmatrix} w_1 & w_2 & w_3 \end{pmatrix}

    And w_2, w_3 are the proportions of tissue-hinderd and free water
    respectively. See below for the estimation proceure
    
    Parameters
    ----------

    tissue_fraction: Full path to a file containing the TF, registered to the
    DWI data and resampled to the DWI data resolution.

    """

    def __init__(self,
                 tissue_fraction,
                 data,
                 bvecs,
                 bvals,
                 alpha1,
                 alpha2,
                 water_D=3,
                 gray_D=1,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 verbose=True):
        
        # Initialize the super-class:
        CanonicalTensorModel.__init__(self,
                                      data,
                                      bvecs,
                                      bvals,
                                      params_file=params_file,
                                      axial_diffusivity=axial_diffusivity,
                                      radial_diffusivity=radial_diffusivity,
                                      affine=affine,
                                      mask=mask,
                                      scaling_factor=scaling_factor,
                                      sub_sample=sub_sample,
                                      over_sample=over_sample,
                                      verbose=verbose)

        self.tissue_fraction = ni.load(tissue_fraction).get_data()

        # Convert the diffusivity constants to signal attenuation:
        self.gray_D = np.exp(-self.bvals[self.b_idx][0] * gray_D)
        self.water_D = np.exp(-self.bvals[self.b_idx][0] * water_D)

        # We're going to grid-search over these:
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
    @desc.auto_attr
    def _flat_tf(self):
        """
        Flatten the TF

        """
        return self.tissue_fraction[self.mask]


    @desc.auto_attr
    def signal(self):
        """
        The relevant signal here is:

        .. math::

           \begin{pmatrix} \frac{S_1}{S^0_1} \\ \frac{S_2}{S^0_2} \\ ... \\
           \frac{S_3}{S^0_3} \\ TF \end{pmatrix} 
        
        """
        dw_signal = self.data[...,self.b_idx].squeeze()
        tf_signal = np.reshape(self.tissue_fraction,
                               self.tissue_fraction.shape + (1,))

        return np.concatenate([dw_signal, tf_signal], -1)

    @desc.auto_attr
    def relative_signal(self):
        """
        The signal attenuation in each b-weighted volume, relative to the mean
        of the non b-weighted volumes. We add the original TF here as a last
        volume, so that we can compare fit to signal. 

        Note
        ----
        Need to overload this function for this class, so that the TF does not
        get attenuated.  

        """
        dw_att= self.data[...,self.b_idx]/np.reshape(self.S0,
                                                       (self.S0.shape + (1,)))

        tf_signal = np.reshape(self.tissue_fraction,
                               self.tissue_fraction.shape + (1,))

        return np.concatenate([dw_att, tf_signal], -1) 

    @desc.auto_attr
    def model_params(self):
        """
        Fitting the weights for the TissueFractionModel is done as a second
        stage, after done fitting the CanonicalTensorModel.
        
        The logic is as follows:

        The isotropic weight calculated in the previous stage subsumes two
        different components: one is the free water isotropic component and the
        other is a hindered tissue water component.

        .. math::

            \w_{iso} = \w_2 D_g + \w_3 D_{csf}
            
        Where $\w_{iso}$ is the weight for the isotropic component fit for
        the initial fit and $\w_{2,3}$ are the weights of tissue water and
        free water respectively. $D_g \approx 1$ and $D_{csf} \approx 3$ are
        the diffusivities of gray and white matter, respectively. 

        In addition, we know that the tissue water, together with the tensor
        signal should account for the tissue fraction measurement:

        .. math::
        
            TF = \w_1 * \lambda_1 + \w_2 * \lambda_2 

        Where $\w_1$ is the weight for the canonical tensor found in
        CanonicalTensorModel and $\w_2$ is the weight on the tissue isotropic
        component. $\lambda_{1,2}$ are additional relative weights of the two
        components within the tissue  (canonical tensor and tissue
        water). Implicitly, $\lambda_3 = 0$, reflecting the fact that the free
        water is not part of the tissue fraction at all. To find \lambda{i}, we
        perform a grid search over plausible values of these and choose the
        ones that best account for the diffusion and TF signal.

        To find $\w_2$ and $\w_3$, we follow these steps:

        0. We find $\w_1 = \w_{tensor}$ using the CanonicalTensorModel
        
        1. We fix the values of \lambda_1 and \lambda_2 and solve for \w_2:

            \w_2 = \frac{TF - \lambda_1 \w_1}{\lambda2} =

        2. From the first equation above, we can then solve for \w_3:

            \w_3 = 1 - \w_{iso} - \w_2
            
        3. We go back to the expanded model and predict the diffusion and the
        TF data for these values of     

        """

        # Start by getting the params for the underlying CanonicalTensorModel:
        temp_p_file = self.params_file
        self.params_file = params_file_resolver(self,
                                                'CanonicalTensorModel')
        tensor_params = super(TissueFractionModel, self).model_params
        
        # Restore order: 
        self.params_file = temp_p_file

        # The tensor weight is the second parameter in each voxel: 
        w_ten = tensor_params[self.mask, 1]
        # And the isotropic weight is the third:
        w_iso = tensor_params[self.mask, 2]

        w2 = (self._flat_tf - self.alpha1 * w_ten) / self.alpha2
        w3 = (1 - w_ten - w2)

        w2_out = ozu.nans(self.shape[:3])
        w3_out = ozu.nans(self.shape[:3])

        w2_out[self.mask] = w2
        w3_out[self.mask] = w3

        # Return tensor_idx, w1, w2, w3 
        return tensor_params[...,0],tensor_params[...,1], w2_out, w3_out

    
    @desc.auto_attr
    def fit(self):
        """
        Derive the fit of the TissueFractionModel
        """
        if self.verbose:
            print("Predicting signal from TissueFractionModel")

        out_flat = np.empty((self._flat_signal.shape[0],
                            self._flat_signal.shape[1] + 1))

        flat_ten_idx = self.model_params[0][self.mask]
        flat_w1 = self.model_params[1][self.mask]
        flat_w2 = self.model_params[2][self.mask]
        flat_w3 = self.model_params[3][self.mask]

        for vox in xrange(out_flat.shape[0]):
            if ~np.any(np.isnan([flat_w1[vox], flat_w2[vox], flat_w3[vox]])):

                ten = (flat_w1[vox] *
                    np.hstack([self.rotations[flat_ten_idx[vox]], self.alpha1]))

                tissue_water = flat_w2[vox] * np.hstack(
                [self.gray_D * np.ones(self._flat_signal.shape[-1]) ,
                                                      self.alpha2])

                free_water = flat_w3[vox] * np.hstack(
                [self.water_D * np.ones(self._flat_signal.shape[-1]) , 0])
                
                # recover the signal:
                out_flat[vox]= ((ten + tissue_water + free_water) *
                                self._flat_S0[vox])

                # But not for the last item, which doesn't need to be
                # multiplied by S0: 
                out_flat[vox][-1]/=self._flat_S0[vox]
            else:
                out_flat[vox] = np.nan
                
        out = ozu.nans(self.signal.shape)
        out[self.mask] = out_flat

        return out


    @desc.auto_attr
    def RMSE(self):
        """
        We need to overload this to make the shapes to broadcast into make
        sense. XXX Need to consider whether it makes sense to take out our
        overloaded signal and relative_signal above, so we might not need this
        either... 
        """
        out = ozu.nans(self.signal.shape[:3])
        flat_fit = self.fit[self.mask][:,:self.fit.shape[-1]-1]
        flat_rmse = ozu.rmse(self._flat_signal, flat_fit)                
        out[self.mask] = flat_rmse
        return out
