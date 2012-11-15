class MultiCanonicalTensorModel(CanonicalTensorModel):
    """
    This model extends CanonicalTensorModel with the addition of another
    canonical tensor. The logic is similar, but the fitting is done for every
    commbination of sphere + n canonical tensors (where n can be set to any
    number > 1, but can't really realistically be estimated for n>2...).
    
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 verbose=True,
                 mode='relative_signal',
                 n_canonicals=2):
        """
        Initialize a MultiCanonicalTensorModel class instance.
        """
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
                                      mode=mode,
                                      verbose=verbose)
        
        self.n_canonicals = n_canonicals

    @desc.auto_attr
    def rot_idx(self):
        """
        The indices into combinations of rotations of the canonical tensor,
        according to the order we will use them in fitting
        """
        # Use stdlib magic to make the indices into the basis set: 
        pre_idx = itertools.combinations(range(self.rot_vecs.shape[-1]),
                                         self.n_canonicals)

        # Generate all of them and store, so you know where you stand
        rot_idx = []
        for i in pre_idx:
            rot_idx.append(i)

        return rot_idx

    @desc.auto_attr
    def ols(self):
        """
        Compute the design matrices the matrices for OLS fitting and the OLS
        solution. Cache them for reuse in each direction over all voxels.
        """
        ols_weights = np.empty((len(self.rot_idx),
                                self.n_canonicals + 1,
                                self._flat_signal.shape[0]))

        iso_regressor, tensor_regressor, fit_to = self.regressors

        where_are_we = 0
        for row, idx in enumerate(self.rot_idx):                
        # 'row' refers to where we are in ols_weights
            if self.verbose:
                if idx[0]==where_are_we:
                    s = "Starting MultiCanonicalTensorModel fit"
                    s += " for %sth set of basis functions"%(where_are_we) 
                    print (s)
                    where_are_we += 1
            # The 'design matrix':
            d = np.vstack([[tensor_regressor[i] for i in idx],
                           iso_regressor]).T
            # This is $(X' X)^{-1} X':
            ols_mat = ozu.ols_matrix(d)
            # Multiply to find the OLS solution:
            ols_weights[row] = np.array(np.dot(ols_mat, fit_to)).squeeze()

        return ols_weights

    @desc.auto_attr
    def model_params(self):
        """
        The model parameters.

        Similar to the CanonicalTensorModel, if a fit has ocurred, the data is
        cached on disk as a nifti file 

        If a fit hasn't occured yet, calling this will trigger a model fit and
        derive the parameters.

        In that case, the steps are as follows:

        1. Perform OLS fitting on all voxels in the mask, with each of the
           $\vec{b}$ combinations, choosing only sets for which all weights are
           non-negative. 

        2. Find the PDD combination that most readily explains the data (highest
           correlation coefficient between the data and the predicted signal)
           That will be the combination used to derive the fit for that voxel.

        """
        # The file already exists: 
        if os.path.isfile(self.params_file):
            if self.verbose:
                print("Loading params from file: %s"%self.params_file)

            # Get the cached values and be done with it:
            return ni.load(self.params_file).get_data()
        else:
            # Looks like we might need to do some fitting... 

            # Get the bvec weights (we don't know how many...) and the
            # isotropic weights (which are always last): 
            b_w = self.ols[:,:-1,:].copy().squeeze()
            i_w = self.ols[:,-1,:].copy().squeeze()

            # nan out the places where weights are negative: 
            b_w[b_w<0] = np.nan
            i_w[i_w<0] = np.nan

            # Weight for each canonical tensor, plus a place for the index into
            # rot_idx and one more slot for the isotropic weight (at the end)
            params = np.empty((self._flat_signal.shape[0],
                               self.n_canonicals + 2))

            if self.verbose:
                print("Fitting MultiCanonicalTensorModel:")
                prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]

            # Find the best OLS solution in each voxel:
            for vox in xrange(self._flat_signal.shape[0]):
                # We do this in each voxel (instead of all at once, which is
                # possible...) to not blow up the memory:
                vox_fits = np.empty((len(self.rot_idx), len(self.b_idx)))
                
                for idx, rot_idx in enumerate(self.rot_idx):
                    # The constant regressor gets added in first:
                    this_relative = i_w[idx,vox] * self.regressors[0][0]
                    # And we add the different canonicals on top of that:
                    this_relative += (np.dot(b_w[idx,:,vox],
                    # The tensor regressors are different in cases where we
                    # are fitting to relative/attenuation signal, so grab that
                    # from the regressors attr:
                    np.array([self.regressors[1][x] for x in rot_idx])))

                    if self.mode == 'relative_signal' or self.mode=='normalize':
                        vox_fits[idx] = this_relative * self._flat_S0[vox]
                    elif self.mode == 'signal_attenuation':
                        vox_fits[idx] = (1 - this_relative) * self._flat_S0[vox]
                
                # Find the predicted signal that best matches the original
                # signal attenuation. That will choose the direction for the
                # tensor we use:
                corrs = ozu.coeff_of_determination(self._flat_signal[vox],
                                                   vox_fits)
                
                idx = np.where(corrs==np.nanmax(corrs))[0]

                # Sometimes there is no good solution:
                if len(idx):
                    # In case more than one fits the bill, just choose the
                    # first one:
                    if len(idx)>1:
                        idx = idx[0]
                    
                    params[vox,:] = np.hstack([idx,
                        np.array([x for x in b_w[idx,:,vox]]).squeeze(),
                        i_w[idx, vox]])
                else:
                    # In which case we set it to all nans:
                    params[vox,:] = np.hstack([np.nan,
                                               self.n_canonicals * (np.nan,),
                                               np.nan])

                if self.verbose: 
                    prog_bar.animate(vox, f_name=f_name)

            # Save the params for future use: 
            out_params = ozu.nans(self.signal.shape[:3]+
                                        (params.shape[-1],))
            out_params[self.mask] = np.array(params).squeeze()
            params_ni = ni.Nifti1Image(out_params, self.affine)
            if self.params_file != 'temp':
                if self.verbose:
                    print("Saving params to file: %s"%self.params_file)
                params_ni.to_filename(self.params_file)

            # And return the params for current use:
            return out_params

    @desc.auto_attr
    def predict_all(self):
        """
        Calculate the predicted signal for all the possible OLS solutions
        """
        # Get the bvec weights (we don't know how many...) and the
        # isotropic weights (which are always last): 
        b_w = self.ols[:,:-1,:].copy().squeeze()
        i_w = self.ols[:,-1,:].copy().squeeze()
        
        # nan out the places where weights are negative: 
        #b_w[b_w<0] = np.nan
        #i_w[i_w<0] = np.nan

        # A predicted signal for each voxel, for each rot_idx, for each
        # direction: 
        flat_out = np.empty((self._flat_signal.shape[0],
                           len(self.rot_idx),
                           self._flat_signal.shape[-1]))

        if self.verbose:
            print("Predicting all signals for MultiCanonicalTensorModel:")
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        for vox in xrange(flat_out.shape[0]):
            for idx, rot_idx in enumerate(self.rot_idx):
                # The constant regressor gets added in first:
                this_relative = i_w[idx,vox] * self.regressors[0][0]
                # And we add the different canonicals on top of that:
                this_relative += (np.dot(b_w[idx,:,vox],
                # The tensor regressors are different in cases where we
                # are fitting to relative/attenuation signal, so grab that
                # from the regressors attr:
                np.array([self.regressors[1][x] for x in rot_idx])))

                if self.mode == 'relative_signal' or self.mode=='normalize':
                    flat_out[vox, idx] = this_relative * self._flat_S0[vox]
                elif self.mode == 'signal_attenuation':
                    flat_out[vox, idx] = (1-this_relative)*self._flat_S0[vox]

            if self.verbose: 
                prog_bar.animate(vox, f_name=f_name)

        out = ozu.nans(self.signal.shape[:3] + 
                       (len(self.rot_idx),) + 
                       (self.signal.shape[-1],))
        out[self.mask] = flat_out

        return out


    @desc.auto_attr
    def fit(self):
        """
        Predict the signal attenuation from the fit of the
        MultiCanonicalTensorModel 
        """

        if self.verbose:
            print("Predicting signal from MultiCanonicalTensorModel")
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]
            
        out_flat = np.empty(self._flat_signal.shape)
        flat_params = self.model_params[self.mask]
        
        for vox in xrange(out_flat.shape[0]):
            # If there's a nan in there, just ignore this voxel and set it to
            # all nans:
            if ~np.any(np.isnan(flat_params[vox, 1])):
                b_w = flat_params[vox,1:1+self.n_canonicals]
                i_w = flat_params[vox,-1]
                # This gets saved as a float, but we can safely assume it's
                # going to be an integer:
                rot_idx = self.rot_idx[int(flat_params[vox,0])]

                out_flat[vox]=(np.dot(b_w,
                               np.array([self.rotations[i] for i in rot_idx])) +
                               self.regressors[0][0] * i_w) * self._flat_S0[vox]
            else:
                out_flat[vox] = np.nan  # This gets broadcast to the right
                                        # length on assigment?
            if self.verbose: 
                prog_bar.animate(vox, f_name=f_name)

        out = ozu.nans(self.signal.shape)
        out[self.mask] = out_flat

        return out

    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        The principal diffusion direction is the direction of the tensor with
        the highest weight
        """
        out_flat = np.empty((self._flat_signal.shape[0] ,3))
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_params[vox][0]):
                # These are the indices into the bvecs:
                idx = [i for i in self.rot_idx[int(flat_params[vox][0])]]
                w = flat_params[vox][1:1+self.n_canonicals]
                # Where's the largest weight:
                out_flat[vox]=\
                    self.bvecs[:,self.b_idx].T[int(idx[np.argsort(w)[-1]])]
                
        out = ozu.nans(self.signal.shape[:3] + (3,))
        out[self.mask] = out_flat
        return out
        
    @desc.auto_attr
    def fit_angle(self):
        """
        The angle between the tensors that were fitted
        """
        out_flat = np.empty(self._flat_signal.shape[0])
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if ~np.isnan(flat_params[vox][0]):
                idx = [i for i in self.rot_idx[int(flat_params[vox][0])]]
                # Sort them according to their weight and take the two
                # weightiest ones:
                w = flat_params[vox,1:1+self.n_canonicals]
                idx = np.array(idx)[np.argsort(w)]
                ang = np.rad2deg(ozu.vector_angle(
                    self.rot_vecs.T[idx[-1]],
                    self.rot_vecs.T[idx[-2]]))

                ang = np.min([ang, 180-ang])
                
                out_flat[vox] = ang
                
            else:
                out_flat[vox] = np.nan

        
        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = out_flat

        return out
