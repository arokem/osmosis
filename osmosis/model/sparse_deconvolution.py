class SparseDeconvolutionModel(CanonicalTensorModel):
    """
    Use the lasso to do spherical deconvolution with a canonical tensor basis
    set. 
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 solver=None,
                 solver_params=None,
                 params_file=None,
                 axial_diffusivity=AD,
                 radial_diffusivity=RD,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 over_sample=None,
                 mode='relative_signal',
                 verbose=True):
        """
        Initialize SparseDeconvolutionModel class instance.
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
        
        # Name the params file, if needed: 
        this_class = str(self.__class__).split("'")[-2].split('.')[-1]
        self.params_file = params_file_resolver(self,
                                                this_class,
                                                params_file=params_file)

        # Deal with the solver stuff: 
        # For now, the default is ElasticNet:
        if solver is None:
            this_solver = sklearn_solvers['ElasticNet']
        # Assume it's a key into the dict: 
        elif isinstance(solver, str):
            this_solver = sklearn_solvers[solver]
        # Assume it's a class: 
        else:
            this_solver = solver
        
        # This will be passed as kwarg to the solver initialization:
        if solver_params is None:
            """
            If you are interested in controlling the L1 and L2 penalty
            separately, keep in mind that this is equivalent to::

                a * L1 + b * L2

            where::

                alpha = a + b and rho = a / (a + b)
            """
            # Taken from Stefan:
            a = 0.0001
            b = 0.00001
            alpha = a + b
            rho = a/(a+b)
            self.solver_params = dict(alpha=alpha,
                                      rho=rho,
                                      fit_intercept=True,
                                      positive=True)
        else:
            self.solver_params = solver_params

        # We reuse the same class instance in all voxels: 
        self.solver = this_solver(**self.solver_params)

    @desc.auto_attr
    def model_params(self):
        """

        Use sklearn to fit the parameters:

        """

        # The file already exists: 
        if os.path.isfile(self.params_file):
            if self.verbose:
                print("Loading params from file: %s"%self.params_file)
            # Get the cached values and be done with it:
            return ni.load(self.params_file).get_data()

        else:

            if self.verbose:
                print("Fitting SparseDeconvolutionModel:")
                prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]

            iso_regressor, tensor_regressor, fit_to = self.regressors

            # One weight for each rotation
            params = np.empty((self._flat_signal.shape[0],
                               self.rotations.shape[0]))

            # We fit the deviations from the mean signal, which is why we also
            # demean each of the basis functions:
            design_matrix = tensor_regressor - np.mean(tensor_regressor, 0)

            # One basis function per column (instead of rows):
            design_matrix = design_matrix.T
            
            for vox in xrange(self._flat_signal.shape[0]):
                # Fit the deviations from the mean of the fitted signal: 
                sig = fit_to.T[vox] - np.mean(fit_to.T[vox])
                # Use the solver you created upon initialization:
                params[vox] = self.solver.fit(design_matrix, sig).coef_
                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)

            out_params = ozu.nans((self.signal.shape[:3] + 
                                          (design_matrix.shape[-1],)))

            out_params[self.mask] = params
            # Save the params to a file: 
            params_ni = ni.Nifti1Image(out_params, self.affine)
            if self.params_file != 'temp':
                if self.verbose:
                    print("Saving params to file: %s"%self.params_file)
                params_ni.to_filename(self.params_file)

            # And return the params for current use:
            return out_params
            

    @desc.auto_attr
    def fit(self):
        """
        Predict the data from the fit of the SparseDeconvolutionModel
        """
        if self.verbose:
            msg = "Predicting signal from SparseDeconvolutionModel"
            msg += " with %s"%self.solver
            print(msg)
        
        iso_regressor, tensor_regressor, fit_to = self.regressors

        design_matrix = tensor_regressor - np.mean(tensor_regressor, 0)
        design_matrix = design_matrix.T
        out_flat = np.empty(self._flat_signal.shape)
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            this_relative = (np.dot(flat_params[vox], design_matrix.T) + 
                            np.mean(fit_to.T[vox]))
            if self.mode == 'relative_signal' or self.mode=='normalize':
                this_pred_sig = this_relative * self._flat_S0[vox]
            elif self.mode == 'signal_attenuation':
                this_pred_sig =  (1 - this_relative) * self._flat_S0[vox]

            # Fit scale and offset:
            #a,b = np.polyfit(this_pred_sig, self._flat_signal[vox], 1)
            # out_flat[vox] = a*this_pred_sig + b
            out_flat[vox] = this_pred_sig 
        out = ozu.nans(self.signal.shape)
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
                idx1 = np.argsort(flat_params[vox])[-1]
                idx2 = np.argsort(flat_params[vox])[-2]
                ang = np.rad2deg(ozu.vector_angle(
                    self.bvecs[:,self.b_idx].T[idx1],
                    self.bvecs[:,self.b_idx].T[idx2]))

                ang = np.min([ang, 180-ang])
                
                out_flat[vox] = ang
                
        else:
            out_flat[vox] = np.nan
        
        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = out_flat

        return out

    @desc.auto_attr
    def odf_peaks(self):
        """
        Calculate the value of the peaks in the ODF (in this case, that is
        defined as the weights on the model params 
        """
        faces = sphere.Sphere(xyz=self.bvecs[:,self.b_idx].T).faces
        odf_flat = self.model_params[self.mask]
        out_flat = np.zeros(odf_flat.shape)
        for vox in xrange(odf_flat.shape[0]):
            peaks, inds = recspeed.local_maxima(odf_flat[vox], faces)
            out_flat[vox][inds] = peaks 

        out = np.zeros(self.model_params.shape)
        out[self.mask] = out_flat
        return out


    @desc.auto_attr
    def n_peaks(self):
        """
        How many peaks in the ODF of each voxel
        """
        return np.sum(self.odf_peaks > 0, -1)


    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        Gives you not only the principal, but also the 2nd, 3rd, etc
        """
        out_flat = ozu.nans(self._flat_signal.shape + (3,))
        flat_params = self.model_params[self.mask]
        for vox in xrange(out_flat.shape[0]):
            coeff_idx = np.where(flat_params[vox]>0)[0]
            for i, idx in enumerate(coeff_idx):
                out_flat[vox, i] = self.bvecs[:,self.b_idx].T[idx]

        
        out = ozu.nans(self.signal.shape + (3,))
        out[self.mask] = out_flat
            
        return out

        
        
    def quantitative_anisotropy(self, Np):
        """

        Return the relative size and indices of the Np major param values
        (canonical tensor) weights  in the ODF 
        """
        if self.verbose:
            print("Calculating quantitative anisotropy:")
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]


        # Allocate space for Np QA values and indices in the entire volume:
        qa_flat = np.zeros((self._flat_signal.shape[0], Np))
        inds_flat = np.zeros(qa_flat.shape, np.int)  # indices! 
        
        flat_params = self.model_params[self.mask]
        for vox in xrange(flat_params.shape[0]):
            this_params = flat_params[vox]
            ii = np.argsort(this_params)[::-1]  # From largest to smallest
            inds_flat[vox] = ii[:Np]
            qa_flat[vox] = (this_params/np.sum(this_params))[inds_flat[vox]] 

            if self.verbose:
                prog_bar.animate(vox, f_name=f_name)

        qa = np.zeros(self.signal.shape[:3] + (Np,))
        qa[self.mask] = qa_flat
        inds = np.zeros(qa.shape)
        inds[self.mask] = inds_flat
        return qa, inds
