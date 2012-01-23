# Microtrack
## Predicting diffusion weighted MRI data from fibers

This is a set of algorithms for 'closing the loop' in diffusion-weighted MRI (DWI), going from sets of fibers to the predictions

The steps taken in this process are:

1. Calculate the expected fiber contributions to the DWI data, based on biophysical models of the fibers and surrounding tissue. 
2. Construct a linear model based on the various contributions to the DWI data. 
3. Solve the linear model with a defined set of constraints. If needed, solve in stages for different components of the biophysical model (fiber component, isotropic tissue component, etc.)
4. Compare the model predictions to the DWI data. Compare different models.   


Microtrack is copyright of the [VISTA lab](http://white.stanford.edu/) at Stanford University and is released under the terms of the [GPL license](http://www.gnu.org/copyleft/gpl.html).  
