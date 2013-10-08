Using Osmosis
=========

Installation
---------

To install :mod:`osmosis` on your computer, grab the code from github. Then, in
the terminal set your pwd to be the top-level directory of the source tree and,
at the terminal, issue:

    python setup.py install


Test your installation
-----------------
You can run the full test-suite by issuing the following from a terminal:

    python -c"import osmosis; osmosis.test()"

Let [me](http://arokem.org/) know if this is not working for you and we will
fix it. Make sure that you have all the
[dependencies](https://github.com/vistalab/osmosis/blob/master/doc/dependencies.rst)
installed. One easy way to get most of these dependencies installed is to use
[Enthought Canopy](https://www.enthought.com/products/canopy/).


Where you can find some data
------------------------
Download all the files from
[here](https://sdr.stanford.edu/items/druid:ng782rw8378).  Place them inside of
a directory. At the beginning of each analysis script, you will want to adjust
the data path to point to the location where you put the files, by setting the
internal `data_path` variable of the :mod:`io` module: 

    import osmosis.io as oio
    oio.data_path = 'path/to/files'

Once you do that, the functions in that module should do your bidding on these
data-sets.

How to get started with analysis
--------------------------

Examples of analysis using osmosis are in the `doc/paper_figures`
directory. These scripts read data from file, fit models and perform evaluation
and comparison of these models.


Modules
-------
- :mod:`boot` : Functions for resampling diffusion data.
- :mod:`cluster` : Functions for spherical clustering.
- :mod:`descriptors` : Helper functions for making turbo-charged classes.
- :mod:`fibers` : Handling fibers and representing them.
- :mod:`io` : File input/output.
- :mod:`leastsqbound` : Bounded least-squares optimization.
- :mod:`model` : Models of diffusion.
- :mod:`model.analysis` : Analysis functions that apply to all models.
- :mod:`model.base` : Base classes for model representation
- :mod:`model.calibrated_tensor` : Model with calibration of the weights on an ROI
- :mod:`model.canonical_tensor` : Simple single stick model
- :mod:`model.csd` : Evaluate mrTrix CSD models.
- :mod:`model.dti` : Diffusion Tensor Imaging.
- :mod:`model.fiber` : An implementation of "LIFE".
- :mod:`model.fiber_tissue` : Use fiber-tracking to find consistent diffusion
  properties.
- :mod:`model.io` : File handling for models.
- :mod:`model.sparse_deconvolution` : SFM.
- :mod:`tensor` :  Abstract representation of tensors and their responses.
- :mod:`utils` :  Functions of generaly utility.
- :mod:`viz` :  Data visualization
- :mod:`volume` :  Handling data in volumes (registration, etc.)







