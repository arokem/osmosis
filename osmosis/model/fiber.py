
import inspect

import numpy as np
import scipy.sparse as sparse

import sklearn.linear_model as lm

import osmosis.utils as ozu
import osmosis.descriptors as desc
import osmosis.sgd as sgd
from osmosis.model.base import BaseModel, SCALE_FACTOR
from osmosis.model.canonical_tensor import AD,RD


def _tensors_from_fiber(f, bvecs, bvals, ad, rd):
    """
    Helper function to get the tensors for each fiber
    """
    return f.tensors(bvecs, bvals, ad, rd)

class BaseFiber(BaseModel):
    """
    Base class for different models that deal with fibers
    """
    def __init__(self,
                 data,
                 FG,
                 bvecs=None,
                 bvals=None,
                 affine=None,
                 mask=None,
		 scaling_factor=None,
		 params_file=None,
		 sub_sample=None):
	"""
	Parameters
	----------
        
	data : a volume with data (can be diffusion data, but doesn't have to
	be)

	FG : a osmosis.fibers.FiberGroup object, or the name of a pdb file
        containing the fibers to be read in using ozf.fg_from_pdb
        """
        # Initialize the super-class:
        BaseModel.__init__(self,
                            data,
                            bvecs,
                            bvals,
                            affine=affine,
                            mask=mask,
                            scaling_factor=scaling_factor,
                            params_file=params_file,
                            sub_sample=sub_sample)

        # The FG is transformed through the provided affine if need be: 
        self.FG = FG.xform(np.dot(FG.affine,self.affine.getI()), inplace=False)

	
    @desc.auto_attr
    def fg_idx(self):
        """
        Indices into the coordinates of the fiber-group
        """
        return self.fg_coords.astype(int)

    
    @desc.auto_attr
    def fg_coords(self):
        """
        All the coords of all the fibers  
        """
        return self.FG.coords


    @desc.auto_attr
    def fg_idx_unique(self):
        """
        The *unique* voxel indices
        """
        return ozu.unique_rows(self.fg_idx.T).T


    @desc.auto_attr
    def voxel2fiber(self):
        """
        The first list in the tuple answers the question: Given a voxel (from
        the unique indices in this model), which fibers pass through it?

        The second answers the question: Given a voxel, for each fiber, which
        nodes are in that voxel? 
        """
        # Preallocate for speed:
        
        # Make a voxels by fibers grid. If the fiber is in the voxel, the value
        # there will be 1, otherwise 0:
        v2f = np.zeros((len(self.fg_idx_unique.T), len(self.FG.fibers)))

        # This is a grid of size (fibers, maximal length of a fiber), so that
        # we can capture put in the voxel number in each fiber/node combination:
        v2fn = ozu.nans((len(self.FG.fibers),
                         np.max([f.coords.shape[-1] for f in self.FG])))

        if self.verbose:
            prog_bar = ozu.ProgressBar(self.FG.n_fibers)
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        # In each fiber:
        for f_idx, f in enumerate(self.FG.fibers):
            # In each voxel present in there:
            for vv in f.coords.astype(int).T:
                # What serial number is this voxel in the unique fiber indices:
                voxel_id = np.where((vv[0] == self.fg_idx_unique[0]) *
                                    (vv[1] == self.fg_idx_unique[1]) *
                                    (vv[2] == self.fg_idx_unique[2]))[0]
                # Add that combination to the grid:
                v2f[voxel_id, f_idx] += 1 
                # All the nodes going through this voxel get its number:
                v2fn[f_idx][np.where((f.coords.astype(int)[0]==vv[0]) *
                                     (f.coords.astype(int)[1]==vv[1]) *
                                     (f.coords.astype(int)[2]==vv[2]))]=voxel_id
            
            if self.verbose:
                prog_bar.animate(f_idx, f_name=f_name)

        return v2f,v2fn

class FiberModel(BaseFiber):
    """
    
    A class for representing and solving predictive models based on
    tractography solutions.
    
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 FG,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 mode='relative_signal',
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):
        """
        Parameters
        ----------
        
        FG: a osmosis.fibers.FiberGroup object, or the name of a pdb file
            containing the fibers to be read in using ozf.fg_from_pdb

        axial_diffusivity: The axial diffusivity of a single fiber population.

        radial_diffusivity: The radial diffusivity of a single fiber population.
        
        """
        # Initialize the super-class:
        BaseFiber.__init__(self,
			   data,
			   FG,
			   bvecs,
			   bvals,
			   affine=affine,
			   mask=mask,
			   scaling_factor=scaling_factor,
			   params_file=params_file,
		           sub_sample=sub_sample)

        self.axial_diffusivity = axial_diffusivity
        self.radial_diffusivity = radial_diffusivity
        self.mode = mode

    @desc.auto_attr
    def fiber_signal(self):
        """
        The relative signal predicted along each fiber. 
        """

        if self.verbose:
            prog_bar = ozu.ProgressBar(self.FG.n_fibers)
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        sig = []
        for f_idx, f in enumerate(self.FG):
            sig.append(f.predicted_signal(self.bvecs[:, self.b_idx],
                                          self.bvals[self.b_idx],
                                          self.axial_diffusivity,
                                          self.radial_diffusivity))

            if self.verbose:
                prog_bar.animate(f_idx, f_name=f_name)

        return sig
        
    @desc.auto_attr
    def matrix(self):
        """
        The matrix of fiber-contributions to the DWI signal.
        """
        # Assign some local variables, for shorthand:
        vox_coords = self.fg_idx_unique.T
        n_vox = self.fg_idx_unique.shape[-1]
        n_bvecs = self.b_idx.shape[0]
        v2f,v2fn = self.voxel2fiber

        # How many fibers in each voxel (this will determine how many
        # components are in the fiber part of the matrix):
        n_unique_f = np.sum(v2f)        
        
        # Preallocate these, which will be used to generate the two sparse
        # matrices:

        # This one will hold the fiber-predicted signal
        f_matrix_sig = np.zeros(n_unique_f * n_bvecs)
        f_matrix_row = np.zeros(n_unique_f * n_bvecs)
        f_matrix_col = np.zeros(n_unique_f * n_bvecs)

        # And this will hold weights to soak up the isotropic component in each
        # voxel: 
        i_matrix_sig = np.zeros(n_vox * n_bvecs)
        i_matrix_row = np.zeros(n_vox * n_bvecs)
        i_matrix_col = np.zeros(n_vox * n_bvecs)
        
        keep_ct1 = 0
        keep_ct2 = 0

        if self.verbose:
            prog_bar = ozu.ProgressBar(len(vox_coords))
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        # In each voxel:
        for v_idx, vox in enumerate(vox_coords):
            # For each fiber:
            for f_idx in np.where(v2f[v_idx])[0]:
                # Sum the signal from each node of the fiber in that voxel: 
                pred_sig = np.zeros(n_bvecs)
                for n_idx in np.where(v2fn[f_idx]==v_idx)[0]:
                    relative_signal = self.fiber_signal[f_idx][n_idx]
                    if self.mode == 'relative_signal':
                        # Predict the signal and demean it, so that the isotropic
                        # part can carry that:
                        pred_sig += (relative_signal -
                            np.mean(self.relative_signal[vox[0],vox[1],vox[2]]))
                    elif self.mode == 'signal_attenuation':
                        pred_sig += ((1 - relative_signal) -
                        np.mean(1 - self.relative_signal[vox[0],vox[1],vox[2]]))
                    
            # For each fiber-voxel combination, we now store the row/column
            # indices and the signal in the pre-allocated linear arrays
            f_matrix_row[keep_ct1:keep_ct1+n_bvecs] =\
                np.arange(n_bvecs) + v_idx * n_bvecs
            f_matrix_col[keep_ct1:keep_ct1+n_bvecs] = np.ones(n_bvecs) * f_idx 
            f_matrix_sig[keep_ct1:keep_ct1+n_bvecs] = pred_sig
            keep_ct1 += n_bvecs

            # Put in the isotropic part in the other matrix: 
            i_matrix_row[keep_ct2:keep_ct2+n_bvecs]=\
                np.arange(v_idx*n_bvecs, (v_idx + 1)*n_bvecs)
            i_matrix_col[keep_ct2:keep_ct2+n_bvecs]= v_idx * np.ones(n_bvecs)
            i_matrix_sig[keep_ct2:keep_ct2+n_bvecs] = 1
            keep_ct2 += n_bvecs
            if self.verbose:
                prog_bar.animate(v_idx, f_name=f_name)
        
        # Allocate the sparse matrices, using the more memory-efficient 'csr'
        # format: 
        fiber_matrix = sparse.coo_matrix((f_matrix_sig,
                                       [f_matrix_row, f_matrix_col])).tocsr()
        iso_matrix = sparse.coo_matrix((i_matrix_sig,
                                       [i_matrix_row, i_matrix_col])).tocsr()

        if self.verbose:
            print("Generated model matrices")

        return (fiber_matrix, iso_matrix)

        
    @desc.auto_attr
    def voxel_signal(self):
        """        
        The signal in the voxels corresponding to where the fibers pass through.
        """
        if self.mode == 'relative_signal':
            return self.relative_signal[self.fg_idx_unique[0],
                                        self.fg_idx_unique[1],
                                        self.fg_idx_unique[2]]

        elif self.mode == 'signal_attenuation':
            return self.signal_attenuation[self.fg_idx_unique[0],
                                           self.fg_idx_unique[1],
                                           self.fg_idx_unique[2]]

    @desc.auto_attr
    def voxel_signal_demeaned(self):
        """        
        The signal in the voxels corresponding to where the fibers pass
        through, with mean removed
        """
        # Get the average, broadcast it back to the original shape and demean,
        # finally ravel again: 
        return(self.voxel_signal.ravel() -
               (np.mean(self.voxel_signal,-1)[np.newaxis,...] +
        np.zeros((len(self.b_idx),self.voxel_signal.shape[0]))).T.ravel())

    
    @desc.auto_attr
    def iso_weights(self):
        """
        Get the weights using scipy.sparse.linalg or sklearn.linear_model.sparse

        """

        iso_w =sgd.stochastic_gradient_descent(self.voxel_signal.ravel(),
                                               self.matrix[1],
                                               verbose=self.verbose)
        
        return iso_w
    
    @desc.auto_attr
    def fiber_weights(self):
        """
        Get the weights for the fiber part of the matrix
        """
        fiber_w = sgd.stochastic_gradient_descent(self.voxel_signal_demeaned,
                                                  self.matrix[0],
                                                  verbose=self.verbose)

        return fiber_w

    
    @desc.auto_attr
    def _fiber_fit(self):
        """
        This is the fit for the non-isotropic part of the signal:
        """
        # return self._Lasso.predict(self.matrix[0])
        return sgd.spdot(self.matrix[0], self.fiber_weights)


    @desc.auto_attr
    def _iso_fit(self):
        # We want this to have the size of the original signal which is
        # (n_bvecs * n_vox), so we broadcast across directions in each voxel:
        return (self.iso_weights[np.newaxis,...] +
                np.zeros((len(self.b_idx), self.iso_weights.shape[0]))).T.ravel()


    @desc.auto_attr
    def fit(self):
        """
        The predicted signal from the FiberModel
        """
        # We generate the SGD prediction and in each voxel, we add the
        # offset, according to the isotropic part of the signal, which was
        # removed prior to fitting:
        
        return np.array(self._fiber_fit + self._iso_fit).squeeze()


class FiberStatistic(BaseFiber):
    """
    
    A class for calculating a fiber statistic based on an additional volume of
    data also provided, according to the idea that the signal y (this can be
    any kind of 3D volume, for example the mean diffusivity from a tensor
    model, or a measurement of MTV, or something like that) can be described in
    each voxel through which fibers pass as a linear combination of that
    statistic along the entire length of each of the fibers. That is, 

    y = Aw

    Where A is the binary v2f matrix, that contains 0's everywhere and 1's
    in the elements representing the voxels (rows) for which that fiber
    (columns) passes through.

    Solving for w provides the value of the statistic along the entire length
    of the fiber, given all the other fibers in the FG and the constraints they
    imply    
    """
    def __init__(self,
                 data,
                 FG,
                 affine=None,
                 mask=None):
        """
        Parameters
        ----------
        
        FG: a osmosis.fibers.FiberGroup object, or the name of a pdb file
            containing the fibers to be read in using ozf.fg_from_pdb        

        data: array, or path to nifti
        """
        # Initialize the super-class:
        BaseFiber.__init__(self,
			   data,
			   FG,
			   affine=affine,
			   mask=mask)

    @desc.auto_attr
    def design_matrix(self):
        """
        The design matrix based on the fiber coordinates
        """
        v2f, v2fn = self.voxel2fiber
        # Binarize this sucker:
        v2f = np.array(v2f, dtype=bool).astype(int)
        # We add a column to account for non-fiber stuff: 
        return np.hstack([v2f, np.eye(v2f.shape[0])])

    
    @desc.auto_attr
    def fiber_data(self):
        """

        """
        return self.data[self.fg_idx_unique[0],
                         self.fg_idx_unique[1],
                         self.fg_idx_unique[2]]


    @desc.auto_attr
    def model_params(self):
        """
        The weights on the fibers calculated from the linear model
        """
        L = lm.ElasticNet(l1_ratio=0.01, alpha=0.0005, positive=True)
        L.fit(self.design_matrix, self.fiber_data)
        return L.coef_, L.intercept_
        #return (ozu.ols_matrix(self.design_matrix)).dot(self.fiber_data)


    @desc.auto_attr
    def coef(self):
        """
        """
        return self.model_params[0]


    @desc.auto_attr
    def intercept(self):
        """
        """
        return self.model_params[1]

    
    @desc.auto_attr
    def fit(self):
        """
        Predict back the data based on the fiber weights
        """
        return(self.coef.dot(self.design_matrix.T) + self.intercept)

        


	
    
