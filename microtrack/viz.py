"""

Visualization functions.

This might eventually call to fvtk for visualization of 3-d stuff in 3-d. For
now, only matplotlib 2-d stuff. 

"""

import matplotlib.pyplot as plt
import numpy as np

def mosaic(vol, fig=None, title=None, size=None, vmin=None, vmax=None):
    """
    Display a 3-d volume of data as a 2-d mosaic
    
    """
    sq = int(np.ceil(np.sqrt(len(vol))))

    # Take the first one, so that you can assess what shape the rest should be: 
    im = np.hstack(vol[0:sq])
    height = im.shape[0]
    width = im.shape[1]
    for i in range(sq):
        this_im = np.hstack(vol[9*i:9*(i+1)])
        wid_margin = width - this_im.shape[-1]
        if wid_margin: 
            this_im = np.hstack([this_im, np.zeros((height, wid_margin))])
        im = np.vstack([im, this_im])
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    imax = ax.matshow(im, vmin=vmin, vmax=vmax)
    ax.get_axes().get_xaxis().set_visible(False)
    ax.get_axes().get_yaxis().set_visible(False)
    cbar = fig.colorbar(imax, ticks=[0, np.nanmax(im)], format='%1.2f')  
    if title is not None:
        ax.set_title("Kappa for %s directions"%n[n_idx])
    if size is not None: 
        fig.set_size_inches([20,12])
