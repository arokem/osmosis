"""

Visualization functions.

This might eventually call to fvtk for visualization of 3-d stuff in 3-d. For
now, only matplotlib stuff. 

"""

from itertools import cycle

import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

import dipy.core.geometry as geo

import microtrack.utils as mtu


# make a color-cycle that can be used in plotting stuff: 
color_cycle = cycle(['maroon','red','purple','fuchsia','green',
                     'lime','olive','yellow','navy','blue'])

def mosaic(vol, fig=None, title=None, size=None, vmin=None, vmax=None,
           return_mosaic=False, return_cbar=False, **kwargs):
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
    if vmin is None:
        vmin = np.nanmin(vol)
    if vmax is None:
        vmax = np.nanmax(vol)
        
    sq = int(np.ceil(np.sqrt(len(vol))))

    # Take the first one, so that you can assess what shape the rest should be: 
    im = np.hstack(vol[0:sq])
    height = im.shape[0]
    width = im.shape[1]
    
    for i in range(1,sq):
        this_im = np.hstack(vol[(len(vol)/sq)*i:(len(vol)/sq)*(i+1)])
        wid_margin = width - this_im.shape[-1]
        if wid_margin: 
            this_im = np.hstack([this_im, np.nan *np.ones((height, wid_margin))])
        im = np.vstack([im, this_im])
    
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        # This assumes that the figure was originally created with this
        # function:
        ax = fig.axes[0]

    imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
    ax.get_axes().get_xaxis().set_visible(False)
    ax.get_axes().get_yaxis().set_visible(False)
    # The colorbar will refer to the last thing plotted in this figure
    cbar = fig.colorbar(imax, ticks=[np.nanmin([0, vmin]),
                        vmax - (vmax - vmin)/2,
                        np.nanmin([vmax,np.nanmax(im)])],
                        format='%1.2f')
    
    if title is not None:
        ax.set_title(title)
    if size is not None: 
        fig.set_size_inches(size)

    returns = [fig]
    if return_mosaic: 
        returns.append(im)
    if return_cbar:
        returns.append(cbar)

    # If you are just returning the fig handle, unpack it: 
    if len(returns)==1:
        returns=returns[0]

    return returns
    
def lut_from_cm(cm, n=256):
    """
    Returns the n-sized look-up table for RGB values for a matplotlib colormap
    cm 

    """
    seg = cm._segmentdata
    rgb = ['red', 'green', 'blue']
    lut = []
    for k in rgb:
        this_seg = np.array(seg[k])
        xp = this_seg[:,0]
        fp = this_seg[:,1]
        x = np.linspace(xp[0],xp[-1],n)

        lut.append(np.interp(x, xp, fp))
    
    return np.array(lut).T


def color_from_val(val, min_val=0, max_val=255,
                   cmap_or_lut=matplotlib.cm.RdBu, n=256):
    """
    Provided some value and some maximal value, what is the rgb value in the
    matplotlib cmap (with n items) or a lut (n by 3 or 4, including rgba)
    which corresponds to it. 

    """

    val = np.asarray(val)
    
    if isinstance(cmap_or_lut, matplotlib.colors.LinearSegmentedColormap):
        lut = lut_from_cm(cmap_or_lut, n=n)
    else:
        lut = cmap_or_lut

    if np.iterable(val):
        rgb = np.zeros((val.shape + (lut.shape[-1],)))
        idx = np.where(~np.isnan(val))
        rgb[idx] = lut[(((float(val[idx])-min_val)/(max_val-min_val))*(n-1)).astype(int)]
        return [tuple(this) for this in rgb]
    else:
        rgb = lut[(((float(val)-min_val)/(max_val-min_val))*(n-1)).astype(int)]        
        return tuple(rgb)


def sig_on_sphere(val, theta, phi, fig=None, sphere_dim=100, r_from_val=False,
                  **kwargs):
    """
    Presente values on a sphere.

    Parameters
    ----------
    val: array with data

    theta, phi: co-linear to the array with data, the theta, phi on the circle
        from which the data was taken

    fig: matplotlib figure, optional. Default: make new figure

    sphere_dim: The data will be interpolated into a sphere_dim by sphere_dim
        grid 

    r_from_val: Whether to double-code the values both by color and by the
        distance from the center

    cmap: Specify a matplotlib colormap to use for coloring the data. 

    Additional kwargs can be passed to the matplotlib.pyplot.plot3D command.

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')

    # Get the cmap argument out of your kwargs, or use the default:
    cmap = kwargs.pop('cmap', matplotlib.cm.RdBu)

    u = np.linspace(0, 2 * np.pi, sphere_dim)
    v = np.linspace(0, np.pi, sphere_dim)

    x_inter = 10 * np.outer(np.cos(u), np.sin(v))
    y_inter = 10 * np.outer(np.sin(u), np.sin(v))
    inter_val = np.zeros(x_inter.shape).ravel()
    z_inter = np.outer(np.ones(np.size(u)), np.cos(v))
    grid_r, grid_theta, grid_phi = geo.cart2sphere(x_inter, y_inter, z_inter)
    for idx, this in enumerate(zip(grid_theta.ravel(), grid_phi.ravel())):
        this_theta = np.abs(theta - np.array(this[0]))
        this_phi = np.abs(phi - np.array(this[1]))
        # The closest theta and phi:
        min_idx = np.argmin(this_theta + this_phi)
        inter_val[idx] = val[min_idx]

    # Calculate the values from the colormap that will be used for the
    # face-colors: 
    c = np.array(color_from_val(inter_val,
                                min_val=np.min(val),
                                max_val=np.max(val),
                                cmap_or_lut=cmap))
    
    new_shape = (x_inter.shape + (3,))

    c = np.reshape(c, new_shape)

    if r_from_val:
        # Use the interpolated values to set z: 
        new_x, new_y, new_z = geo.sphere2cart(inter_val.reshape(x_inter.shape),
                                              grid_theta, grid_phi)

        ax.plot_surface(new_x, new_y, new_z, rstride=4, cstride=4,
                        facecolors=c, **kwargs)
    else: 
        ax.plot_surface(x_inter, y_inter, z_inter, rstride=4, cstride=4,
                        facecolors=c, **kwargs)

    return fig

def sig_in_points(bvecs, val, fig=None, r_from_val=False, **kwargs):
    """
    Display signal in points in 3d, based on the bvecs provided.

    Parameters
    ----------
    bvecs: 3 by n
       unit (usually) vectors defining the x,y,z coordinates in which to
       display the data

    val: array of length n, with values of the data to present

    fig: Whether to draw this in an existing figure (assumes that you want to
    put this in fig.axes[0], unless you provide an axes3d.Axes3D class instance
    as an input, in which case, we'll use that instance.

    r_from_val: bool, optional.
        Whether to scale the distance from the center for each point by the
        value of that point. Defaults to False.

    
    
    cmap: Specify a matplotlib colormap to use for coloring the data. 
    
    Additional kwargs can be passed to the matplotlib.pyplot.plot3D command.
    """
    if r_from_val:
        x,y,z = scale_bvecs_by_sig(bvecs, val)
    else:
        x,y,z = bvecs

    if fig is None: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    else:
        # If you gave us axes, we'll use those:
        if isinstance(fig, axes3d.Axes3D):
            ax = fig
        # Otherwise, we'll assume you want it in fig.axes[0]:
        else: 
            ax = fig.axes[0]
    
    # Get the cmap argument out of your kwargs:
    cmap = kwargs.pop('cmap', matplotlib.cm.RdBu)
    
    for idx, this_val in enumerate(val):
        c = color_from_val(this_val, min_val=np.min(val), max_val=np.max(val),
                           cmap_or_lut=cmap)
        
        # plot3D expects something with length, so we convert into 1-item arrays:
        ax.plot3D(np.ones(1) * x[idx], 
                  np.ones(1) * y[idx],
                  np.ones(1) * z[idx],
                  'o', c=c, **kwargs)
    return fig

def scale_bvecs_by_sig(bvecs, sig):
    """
    Helper function to rescale your bvecs according to some signal, so that
    they don't fall on the unit sphere, but instead are represented in space as
    distance from the origin.
    """

    x,y,z = bvecs

    r, theta, phi = geo.cart2sphere(x, y, z)

    # Simply replace r with sig:
    x,y,z = geo.sphere2cart(sig, theta, phi)

    return x,y,z


def scatter_density(x,y, res=100, cmap=matplotlib.cm.hot_r):
    """
    Create a scatter plot with density of the data-points at each point on the
    x,y grid coded by the color in the colormap (hot per default)
    
    """

    max_x = np.nanmax(x)
    max_y = np.nanmax(y)
    min_x = np.nanmin(x)
    min_y = np.nanmin(y)
    
    x = mtu.rescale(x).ravel() * (res - 1) 
    y = mtu.rescale(y).ravel() * (res - 1)

    data_arr = np.zeros((res, res))

    for this_x,this_y in zip(x,y):
        # If one of them is a nan, move on:
        if np.isnan(this_x) or np.isnan(this_y):
            pass
        else: 
            data_arr[np.floor(this_x), np.floor(this_y)] += 1

    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imax = ax.matshow(np.log10(np.flipud(data_arr.T)), cmap=cmap)    
    fig.colorbar(imax)
    ax.set_xticks([0] + [i * res/5.0 for i in range(5)])
    ax.set_yticks([0] + [i * res/5.0 for i in range(5)])
    ax.set_xticklabels([0] + ['%0.2f'%(i * ((max_x - min_x)/5.0) + min_x) for i in range(5)])
    ax.set_yticklabels([0] + ['%0.2f'%(i * ((max_y - min_y)/5.0) + min_y) for i in range(5,0,-1)])
    
    return fig


# XXX Maybe implement the following as a subclass of matplotlib.axes.Axes?

def quick_ax(fig=None,subplot=111):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot)
    return ax

