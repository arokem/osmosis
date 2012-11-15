"""


"""


class SparseKernelModel(BaseModel):
    """

    Stefan VdW's kernel model
    
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 sh_order=8,
                 quad_points=132,
                 verts_level=5,
                 alpha=None,
                 rho=None,
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

        self.sh_order = sh_order
        self.quad_points = quad_points

        # This will soon be replaced by an import from dipy:
        import kernel_model
        from dipy.core.subdivide_octahedron import create_unit_hemisphere
        
        self.kernel_model = kernel_model
        self.verts_level = verts_level

        # Set the sparseness params.
        # First, for the default values:
        aa = 0.0001  # L1 weight
        bb = 0.00001 # L2 weight
        if alpha is None:
            alpha = aa + bb
        if rho is None: 
            rho = aa/(aa + bb)

        self.alpha = alpha
        self.rho = rho
            
    @desc.auto_attr
    def _km(self):
        return self.kernel_model.SparseKernelModel(self.bvals[self.b_idx],
                                                   self.bvecs[:,self.b_idx].T,
                                                   sh_order=self.sh_order,
                                                   qp=self.quad_points,
                                                   loglog_tf=False,
                                                   #alpha=self.alpha,
                                                   #rho=self.rho
                                                   )

    
    @desc.auto_attr
    def model_params(self):
        """
        Fit the parameters of the kernel model
        """

        # The file already exists: 
        if os.path.isfile(self.params_file):
            if self.verbose:
                print("Loading params from file: %s"%self.params_file)
            # Get the cached values and be done with it:
            return ni.load(self.params_file).get_data()
        else:

            if self.verbose:
                print("Fitting params for SparseKernelModel")
                prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
                this_class = str(self.__class__).split("'")[-2].split('.')[-1]
                f_name = this_class + '.' + inspect.stack()[0][3]

            # 1 parameter for each basis function + 1 for the intercept:
            out_flat = np.empty((self._flat_signal.shape[0], self.quad_points+1))
            for vox in xrange(out_flat.shape[0]):
                this_fit = self._km.fit(self._flat_relative_signal[vox])
                beta = this_fit.beta
                intercept = this_fit.intercept
                # Fit the model, get the params:
                out_flat[vox] = np.hstack([intercept, beta])
                
                if self.verbose:
                    prog_bar.animate(vox, f_name=f_name)

            out_params = ozu.nans(self.signal.shape[:3] + (self.quad_points+1,))
            out_params[self.mask] = out_flat
            if self.params_file != 'temp':
                # Save the params for future use: 
                params_ni = ni.Nifti1Image(out_params, self.affine)
                if self.verbose:
                    print("Saving params to file: %s"%self.params_file)
                params_ni.to_filename(self.params_file)

            # And return the params for current use:
            return out_params
                                
    @desc.auto_attr
    def fit(self):
        """
        Predict the signal based on the kernel model fit
        """
        if self.verbose:
            print("Predicting signal from SparseKernelModel")
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]


        out_flat = np.zeros(self._flat_signal.shape)
        flat_params = self.model_params[self.mask]

        # We will use a cached fit object generated in the first iteration
        _fit_obj = None
        # And the vertices are the b vectors:
        _verts = self.bvecs[:,self.b_idx].T
        for vox in xrange(out_flat.shape[0]):
            this_fit = self.kernel_model.SparseKernelFit(
                                    flat_params[vox][1:],
                                    flat_params[vox][0],
                                    model=self._km) 

            this_relative = this_fit.predict(cache=_fit_obj,
                                             vertices=_verts)            
            _fit_obj = this_fit # From now on, we will use this cached object
            _verts = None # And set the verts input to None, so that it is
                          # ignored in future iterations
            
            out_flat[vox] = this_relative * self._flat_S0[vox]

            if self.verbose:
                prog_bar.animate(vox, f_name=f_name)

        out = ozu.nans(self.signal.shape)
        out[self.mask] = out_flat
        return out

    @desc.auto_attr
    def odf_verts(self):
        """
        The vertices on which to estimate the odf
        """
        verts, edges, sides = create_unit_hemisphere(self.verts_level)

        return verts, edges

    @desc.auto_attr
    def odf(self):
        """
        The orientation distribution function estimated from the SparseKernel
        model  
        """
        _verts = self.odf_verts[0] # These are the vertices on which we estimate
                                   # the ODF 
        if self.verbose:
            prog_bar = ozu.ProgressBar(self._flat_signal.shape[0])
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            f_name = this_class + '.' + inspect.stack()[0][3]

        out_flat = np.zeros((self._flat_signal.shape[0], _verts.shape[0]))
        flat_params = self.model_params[self.mask]

        # We are going to use cached computations in the fit object: 
        _fit_obj = None # Initially we don't have a cached fit object
        for vox in xrange(out_flat.shape[0]):
            this_fit = self.kernel_model.SparseKernelFit(
                                                flat_params[vox][1:],
                                                flat_params[vox][0],
                                                model=self._km)
            out_flat[vox] = this_fit.odf(cache=_fit_obj, vertices=_verts)
            _fit_obj = this_fit # From now on, we will use this cached object
            _verts = None # And we need to ignore the vertices, so that the
                          # cached fit object can use the cached computation. 
            
            if self.verbose:
                prog_bar.animate(vox, f_name=f_name)

        out = ozu.nans(self.signal.shape[:3] + (out_flat.shape[-1],))
        out[self.mask] = out_flat
        return out


    @desc.auto_attr
    def fit_angle(self):
        """
        The angle between the two primary peaks in the ODF
        
        """
        out_flat = np.zeros(self._flat_signal.shape[0])
        flat_odf = self.odf[self.mask]
        for vox in xrange(out_flat.shape[0]):
            if np.any(np.isnan(flat_odf[vox])):
                out_flat[vox] = np.nan
            else:
                p, i = recspeed.local_maxima(flat_odf[vox], self.odf_verts[1])
                mask = p > 0.5 * np.max(p)
                p = p[mask]
                i = i[mask]

                if len(p) < 2:
                    out_flat[vox] = np.nan
                else:
                    out_flat[vox] = np.rad2deg(ozu.vector_angle(
                                        self.odf_verts[0][i[0]],
                                        self.odf_verts[0][i[1]]))

        out = ozu.nans(self.signal.shape[:3])
        out[self.mask] = out_flat
        return out
    
    @desc.auto_attr
    def principal_diffusion_direction(self):
        """
        The direction of the primary peak of the ODF
        """

        raise NotImplementedError

