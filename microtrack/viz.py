"""

Visualization functions.

This might eventually call to fvtk for visualization of 3-d stuff in 3-d. For
now, only matplotlib 2-d stuff. 

"""

import matplotlib.pyplot as plt
import numpy as np

def mosaic(vol, fig=None, title=None, size=None, vmin=None, vmax=None, **kwargs):
    """
    Display a 3-d volume of data as a 2-d mosaic

    Parameters
    ----------
    vol: 3-d array
       The data

    fig: matplotlib figure, optional
        If this should appear in an already existing figure instance

    title: str, optional
        Title for the plot

    size: [width, height], optional

    vmin/vmax: upper and lower clip-limits on the color-map

    **kwargs: additional arguments to matplotlib.pyplot.matshow
       For example, the colormap to use, etc.
       
    Returns
    -------
    fig: The figure handle
    
    """
    sq = int(np.ceil(np.sqrt(len(vol))))

    # Take the first one, so that you can assess what shape the rest should be: 
    im = np.hstack(vol[0:sq])
    height = im.shape[0]
    width = im.shape[1]
    
    for i in range(1,sq):
        this_im = np.hstack(vol[(len(vol)/sq)*i:(len(vol)/sq)*(i+1)])
        wid_margin = width - this_im.shape[-1]
        if wid_margin: 
            this_im = np.hstack([this_im, np.zeros((height, wid_margin))])
        im = np.vstack([im, this_im])
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
    ax.get_axes().get_xaxis().set_visible(False)
    ax.get_axes().get_yaxis().set_visible(False)
    cbar = fig.colorbar(imax, ticks=[0, np.nanmin([vmax,np.nanmax(im)])/2.,
                                     np.nanmin([vmax,np.nanmax(im)])],
                                     format='%1.2f')  
    if title is not None:
        ax.set_title(title)
    if size is not None: 
        fig.set_size_inches(size)
