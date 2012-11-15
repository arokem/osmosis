"""

Base classes for the model module.

"""

class DWI(desc.ResetMixin):
    """
    A class for representing dwi data
    """
            
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 verbose=True
                 ):
        """
        Initialize a DWI object

        Parameters
        ----------
        data: str or array
            The diffusion weighted mr data provided either as a full path to a
            nifti file containing the data, or as a 4-d array.

        bvecs: str or array
            The unit vectors describing the directions of data
            acquisition. Either an 3 by n array, or a full path to a text file
            containing the 3 by n data.

        bvals: str or array
            The values of b weighting in the data acquisition. Either a 1 by n
            array, or a full path to a text file containing the values.
        
        affine: optional, 4 by 4 array
            The affine provided in the file can be overridden by explicitely
            setting this input variable. If this is left as None, one of two
            things will happen. If the 'data' input was a file-name, the affine
            will be read from that file. Otherwise, a warning will be issued
            and affine will default to np.eye(4).

        mask: optional, 3-d array
            When provided, used as a boolean mask into the data for access. 

        sub_sample: int or array of ints.
           If we want to sub-sample the DWI data on the sphere (in the bvecs),
           we can do one of two things: 
           
        1. If sub_sample is an integer, that number of random bvecs will be
           chosen from the data.

        2. If an array of indices is provided, these will serve as indices into
        the last dimension of the data and only that part of the data will be
        used

        verbose: boolean, optional.
           Whether or not to print out various messages as you go
           along. Default: True

        """
        self.verbose=verbose
        
        # All inputs are handled essentially the same. Inputs can be either
        # strings, in which case file reads are required, or arrays, in which
        # case no file reads are needed and we assign these arrays into the
        # attributes:
        for name, val in zip(['data', 'bvecs', 'bvals'],
                             [data, bvecs, bvals]): 
            if isinstance(val, str):
                exec("self.%s_file = '%s'"% (name, val))
            elif isinstance(val, np.ndarray):
                # This time we need to give it the name-space:
                exec("self.%s = val"% name, dict(self=self, val=val))
            else:
                e_s = "%s seems to be neither an array, "% name
                e_s += "nor a file-name\n"
                e_s += "The value provided was: %s" % val
                raise ValueError(e_s)

        # You might have to scale the bvalues by some factor, so that the units
        # come out correctly in the adc calculation:
        self.bvals = self.bvals.copy() / scaling_factor
        
        # You can provide your own affine, if you want and that bypasses the
        # class method provided below as an auto-attr:
        if affine is not None:
            self.affine = np.matrix(affine)

        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # If it's a string, assume it's the full-path to a nifti file with
            # a binary mask: 
            if isinstance(mask, str):
                mask = ni.load(mask).get_data()
            self.mask = np.array(mask, dtype=bool)

        else:
            # Spatial mask (take only the spatial dimensions):
            self.mask = np.ones(self.shape[:3], dtype=bool)

        if sub_sample is not None:
            if np.iterable(sub_sample):
                idx = sub_sample
            else:
                idx = boot.subsample(self.bvecs[:,self.b_idx].T, sub_sample)[1]
            
            self.b_idx = self.b_idx[idx]
            # At this point, signal will be taken according to these
            # sub-sampled indices:
            self.data = np.concatenate([self.signal,
                                        self.data[:,:,:,self.b0_idx]],-1)
            
            self.b0_idx = np.arange(len(self.b0_idx))

            self.bvecs = np.concatenate([np.zeros((3,len(self.b0_idx))),
                                        self.bvecs[:, self.b_idx]],-1)

            self.bvals = np.concatenate([np.zeros(len(self.b0_idx)),
                                         self.bvals[self.b_idx]])
            self.b_idx = np.arange(len(self.b0_idx), len(self.b0_idx) + len(idx))


    @desc.auto_attr
    def shape(self):
        """
        Get the shape of the data. If possible, don't even load it from file to
        get that. 
        """
        # It must have been in an array
        if not hasattr(self, 'data_file'):
            # No reason not to refer to it directly:
            return self.data.shape
        
        # The data is in a file, and you might not have loaded it yet:
        else:
            # No need to actually load it yet:
            return ni.load(self.data_file).shape

            
    @desc.auto_attr
    def bvals(self):
        """
        If bvals were not provided as an array, read them from file
        """
        if self.verbose:
            print("Loading from file: %s"%self.bvals_file)

        return np.loadtxt(self.bvals_file)

    
    @desc.auto_attr
    def bvecs(self):
        """
        If bvecs were not provided as an array, read them from file
        """ 
        if self.verbose:
            print("Loading from file: %s"%self.bvecs_file)

        return np.loadtxt(self.bvecs_file)

    @desc.auto_attr
    def data(self):
        """
        Load the data from file
        """
        if self.verbose:
            print("Loading from file: %s"%self.data_file)

        return ni.load(self.data_file).get_data()

    @desc.auto_attr
    def affine(self):
        """
        Get the affine transformation of the data to world coordinates
        (relative to acpc)
        """
        if hasattr(self, 'data_file'):
            # This means that there might be an affine to read in from file.
            return np.matrix(ni.load(self.data_file).get_affine())
        else:
            w_s = "DWI data generated from array. Affine will be set to"
            w_s += " np.eye(4)"
            warnings.warn(w_s)
            return np.matrix(np.eye(4))

    @desc.auto_attr
    def _flat_data(self):
        """
        Get the flat data only in the mask
        """        
        return self.data[self.mask]
               
    @desc.auto_attr
    def _flat_S0(self):
        """
        Get the signal in the b0 scans in flattened form (only in the mask)
        """
        return np.mean(self._flat_data[:,self.b0_idx], -1)


    @desc.auto_attr
    def _flat_signal(self):
        """
        Get the signal in the diffusion-weighted volumes in flattened form
        (only in the mask).
        """
        return self._flat_data[:,self.b_idx]


    def signal_reliability(self,
                           DWI2,
                           correlator=stats.pearsonr,
                           r_idx=0,
                           square=True):
        """
        Calculate the r-squared of the correlator function provided, in each
        voxel (across directions, not including b0) between this class instance
        and another class instance, provided as input.

        Parameters
        ----------
        DWI2: Another DWI class instance, with data that should look the same,
            if there wasn't any noise in the measurement

        correlator: callable. This is a function that calculates a measure of
             correlation (e.g. stats.pearsonr, or stats.linregress)

        r_idx: int,
            points to the location of r within the tuple returned by
            the correlator callable if r_idx is negative, that means that the
            return value is not a tuple and should be treated as the value
            itself.

        square: bool,
            If square is True, that means that the value returned from
            the correlator should be squared before returning it, otherwise,
            the value itself is returned.
            
        """
                
        val = np.empty(self._flat_signal.shape[0])

        for ii in xrange(len(val)):
            if r_idx>=0:
                val[ii] = correlator(self._flat_signal[ii],
                                     DWI2._flat_signal[ii])[r_idx] 
            else:
                val[ii] = correlator(self._flat_signal[ii],
                                     DWI2._flat_signal[ii]) 

        if square:
            if has_numexpr:
                r_squared = numexpr.evaluate('val**2')
            else:
                r_squared = val**2
        else:
            r_squared = val
        
        # Re-package it into a volume:
        out = ozu.nans(self.shape[:3])
        out[self.mask] = r_squared

        out[out<-1]=-1.0
        out[out>1]=1.0

        return out 

    @desc.auto_attr
    def b_idx(self):
        """
        The indices into non-zero b values
        """
        return np.where(self.bvals > 0)[0]
        
    @desc.auto_attr
    def b0_idx(self):
        """
        The indices into zero b values
        """
        return np.where(self.bvals==0)[0]

    @desc.auto_attr
    def S0(self):
        """
        Extract and average the signal for volumes in which no b weighting was
        used (b0 scans)
        """
        return np.mean(self.data[...,self.b0_idx],-1).squeeze()
        
    @desc.auto_attr
    def signal(self):
        """
        The signal in b-weighted volumes
        """
        return self.data[...,self.b_idx].squeeze()

    @desc.auto_attr
    def relative_signal(self):
        """
        The signal in each b-weighted volume, relative to the mean
        of the non b-weighted volumes
        """
        # Need to broadcast for this to work:
        signal_rel = self.signal/np.reshape(self.S0, (self.S0.shape + (1,)))
        # Convert infs to nans:
        signal_rel[np.isinf(signal_rel)] = np.nan
        return signal_rel

    @desc.auto_attr
    def _flat_relative_signal(self):
        """
        Get the flat relative signal only in the mask
        """        
        return np.reshape(self.relative_signal[self.mask],
                          (-1, self.b_idx.shape[0]))


    @desc.auto_attr
    def signal_attenuation(self):
        """
        The amount of attenuation of the signal. This is simply: 

           1-relative_signal 

        """
        return 1 - self.relative_signal

    @desc.auto_attr
    def _flat_signal_attenuation(self):
        """

        """
        return 1-self._flat_relative_signal
    

class BaseModel(DWI):
    """
    Base-class for models.
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None,
                 params_file=None,
                 verbose=True):
        """
        A base-class for models based on DWI data.

        Parameters
        ----------

        scaling_factor: int, defaults to 1000.
           To get the units in the S/T equation right, how much do we need to
           scale the bvalues provided.
        
        """
        # DWI should already have everything we need: 
        DWI.__init__(self,
                         data,
                         bvecs,
                         bvals,
                         affine=affine,
                         mask=mask,
                         scaling_factor=scaling_factor,
                         sub_sample=sub_sample,
                         verbose=verbose)

        # Sometimes you might want to not store the params in a file: 
        if params_file == 'temp':
            self.params_file='temp'
        else:
            # Introspect to figure out what name the current class has:
            this_class = str(self.__class__).split("'")[-2].split('.')[-1]
            self.params_file = params_file_resolver(self,
                                                    this_class,
                                                    params_file=params_file)


    @desc.auto_attr
    def adc(self):
        """
        This is the empirically defined ADC:

        .. math::
        
            ADC = -log \frac{S}{b S0}

        """
        out = ozu.nans(self.signal.shape)
        
        out[self.mask] = ((-1/self.bvals[self.b_idx][0]) *
                        np.log(self._flat_relative_signal))

        return out

    @desc.auto_attr
    def fit(self):
        """
        Each model will have a model prediction, which is always in this class
        method. This prediction is used in other methods, such as 'residuals'
        and 'r_squared', etc.

        In this particular case, we set fit to be exactly equal to the
        signal. This should make testing easy :-) 
        """
        return self.signal

    @desc.auto_attr
    def _flat_fit(self):
        """
        Extract a flattened version of the fit, defined for masked voxels
        """
        
        return self.fit[self.mask].reshape((-1, self.signal.shape[-1])) 
    

    def _correlator(self, correlator, r_idx=0, square=True):
        """
        Helper function that uses a callable "func" to apply between two 1-d
        arrays. These 1-d arrays can have different outputs and the one we
        always want is the one which is r_idx into the output tuple 
        """

        val = np.empty(self._flat_signal.shape[0])

        for ii in xrange(len(val)):
            if r_idx>=0:
                val[ii] = correlator(self._flat_signal[ii],
                                     self._flat_fit[ii])[r_idx] 
            else:
                val[ii] = correlator(self._flat_signal[ii],
                                     self._flat_fit[ii]) 
        if square:
            if has_numexpr:
                r_squared = numexpr.evaluate('val**2')
            else:
                r_squared = val**2
        else:
            r_squared = val
        
        # Re-package it into a volume:
        out = ozu.nans(self.shape[:3])
        out[self.mask] = r_squared

        out[out<-1]=-1.0
        out[out>1]=1.0

        return out 

    @desc.auto_attr
    def r_squared(self):
        """
        The r-squared ('explained variance') value in each voxel
        """
        return self._correlator(stats.pearsonr, r_idx=0)
    
    @desc.auto_attr
    def R_squared(self):
        """
        The R-squared ('coefficient of determination' from a linear model fit)
        in each voxel
        """
        return self._correlator(stats.linregress, r_idx=2)

    @desc.auto_attr
    def coeff_of_determination(self):
        """
        Explained variance as: 100 *(1-\frac{RMS(residuals)}{RMS(signal)})

        http://en.wikipedia.org/wiki/Coefficient_of_determination
        
        """
        return self._correlator(ozu.coeff_of_determination,
                                r_idx=np.nan,
                                square=False)

    @desc.auto_attr
    def RMSE(self):
        """
        The square-root of the mean of the squared residuals
        """

        # Preallocate the output:

        out = ozu.nans(self.data.shape[:3])
        res = self.residuals[self.mask]
        
        if has_numexpr:
            out[self.mask] = np.sqrt(np.mean(
                             numexpr.evaluate('res ** 2'), -1))
        else:
            out[self.mask] = np.sqrt(np.mean(np.power(res, 2), -1))
        
        return out

    
    @desc.auto_attr
    def residuals(self):
        """
        The prediction-subtracted residual in each voxel
        """
        out = ozu.nans(self.signal.shape)
        sig = self._flat_signal
        fit = self._flat_fit
        
        if has_numexpr:
            out[self.mask] = numexpr.evaluate('sig - fit')

        else:
            out[self.mask] = sig - fit
            
        return out

    @desc.auto_attr
    def rRMSE(self):
        """
        This is a measure of goodness of fit, based on a relative RMSE measure
        between the RMSE of the fit residuals and the actual signal and the RMS
        of the (mean-removed) b0 measurements.
    
        """
        
        # Get the RMSE of the model relative to the actual data: 
        rmse_model = np.sqrt(np.mean(self.residuals**2, -1))

        # Normalize that to the variance in the b0, which is an estimate of data
        # reliability:
        rms_b0 = ozu.rms(self.data[...,self.b0_idx]-
                         np.mean(self.S0)[...,np.newaxis])

        return rmse_model/rms_b0


class SphereModel(BaseModel):
    """
    This is a very simple model, where for each direction we predict the
    average signal across directions in that voxel
    """
    def __init__(self,
                 data,
                 bvecs,
                 bvals,
                 params_file=None,
                 affine=None,
                 mask=None,
                 scaling_factor=SCALE_FACTOR,
                 sub_sample=None):

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

    @desc.auto_attr
    def fit(self):
        """
        Just calculate the mean and broadcast it to all directions 
        """
        mean_sig = np.mean(self.signal, -1)
        return (mean_sig.reshape(mean_sig.shape + (1,)) +
                np.zeros(mean_sig.shape + (len(self.b_idx),)))
