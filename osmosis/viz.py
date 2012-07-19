"""

Visualization functions.


"""

import sys, time
from itertools import cycle

try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

try:
    from mayavi import mlab as maya
    have_maya = True
except ImportError:
    have_maya = False
    
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.interpolate as interp

import numpy as np

import dipy.core.geometry as geo
from dipy.data import get_sphere
from dipy.utils.spheremakers import sphere_vf_from
from dipy.core.triangle_subdivide import (create_unit_sphere,
                                          create_half_unit_sphere)
from dipy.core.sphere import Sphere, interp_rbf


import osmosis.utils as mtu
import osmosis.tensor as ozt

# make a color-cycle that can be used in plotting stuff: 
color_cycle = cycle(['maroon','red','purple','fuchsia','green',
                     'lime','olive','yellow','navy','blue'])

def mosaic(vol, fig=None, title=None, size=[10,10], vmin=None, vmax=None,
           return_mosaic=False, cbar=True, return_cbar=False, **kwargs):
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

    # If this is a 4D thing and it has 3 as the last dimension
    if len(im.shape)>2:
        if im.shape[2]==3 or im.shape[2]==4:
            mode='rgb'
        else:
            e_s = "This array has too many dimensions for this"
            raise ValueError(e_s)
    else:
        mode = 'standard'
    
    for i in range(1,sq):
        this_im = np.hstack(vol[(len(vol)/sq)*i:(len(vol)/sq)*(i+1)])
        wid_margin = width - this_im.shape[1]
        if wid_margin:
            if mode == 'standard':
                this_im = np.hstack([this_im,
                                     np.nan *np.ones((height, wid_margin))])
            else:
                this_im = np.hstack([this_im,
                                     np.nan * np.ones((im.shape[2],
                                                      height,
                                                      wid_margin))])
        im = np.concatenate([im, this_im], 0)
    
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        # This assumes that the figure was originally created with this
        # function:
        ax = fig.axes[0]

    if mode == 'standard':
        imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
    else:
        imax = plt.imshow(np.rot90(im), interpolation='nearest')
        cbar = False
    ax.get_axes().get_xaxis().set_visible(False)
    ax.get_axes().get_yaxis().set_visible(False)
    returns = [fig]
    if cbar: 
        # The colorbar will refer to the last thing plotted in this figure
        cbar = fig.colorbar(imax, ticks=[np.nanmin([0, vmin]),
                        vmax - (vmax - vmin)/2,
                        np.nanmin([vmax,np.nanmax(im)])],
                        format='%1.2f')
        if return_cbar:
            returns.append(cbar)
    
    if title is not None:
        ax.set_title(title)
    if size is not None: 
        fig.set_size_inches(size)

    if return_mosaic: 
        returns.append(im)

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
        rgb[idx] = lut[((((val[idx]).astype(float)-min_val)/(max_val-min_val))*(n-1)).astype(int)]
        return [tuple(this) for this in rgb]
    else:
        rgb = lut[(((float(val)-min_val)/(max_val-min_val))*(n-1)).astype(int)]        
        return tuple(rgb)


def sig_on_sphere(bvecs, val, fig=None, sphere_dim=1000, r_from_val=False,
                  **kwargs):
    """
    Presente values on a sphere.

    Parameters
    ----------
    bvecs: co-linear to the array with data, the theta, phi on the circle
        from which the data was taken

    val: array with data

    fig: matplotlib figure, optional. Default: make new figure

    sphere_dim: The data will be interpolated into a sphere_dim by sphere_dim
        grid 

    r_from_val: Whether to double-code the values both by color and by the
        distance from the center

    cmap: Specify a matplotlib colormap to use for coloring the data. 

    Additional kwargs can be passed to the matplotlib.pyplot.plot3D command.

    """

    # We don't need the r output
    _, theta, phi = geo.cart2sphere(bvecs[0], bvecs[1], bvecs[2])

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

def sig_in_points(bvecs, val=None, fig=None, r_from_val=False, **kwargs):
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


def scatter_density(x,y, res=100, cmap=matplotlib.cm.hot_r, size=None):
    """
    Create a scatter plot with density of the data-points at each point on the
    x,y grid coded by the color in the colormap (hot per default)
    
    """

    x = np.copy(x)
    y = np.copy(y)
    
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

    if size is not None:
        fig.set_size_inches(size)
    return fig


# XXX Maybe implement the following as a subclass of matplotlib.axes.Axes?

def quick_ax(fig=None,subplot=111):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(subplot)
    return ax


def plot_ellipse_mpl(Dxx, Dyy, Dzz, rstride=4, cstride=4, color='b'):
    """
    Plot an ellipse shape (in contrast to a tensor ellipsoid, see
    plot_ellipsoid_mpl for that) in 3d using matplotlib
    
    """
    
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')

    coefs = Dxx, Dyy, Dzz # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 =

    # 1 Radii corresponding to the coefficients:
    rx, ry, rz = Dxx, Dyy, Dzz#[1/np.sqrt(coef) for coef in coefs]

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=rstride, cstride=cstride, color=color)

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    return fig


class ProgressBar:
    def __init__(self, iterations):
        """
        Progress bar for tracking the progress of long calculations.

        Parameters
        ----------

        iterations: int
            How many iterations does this calculation have before it's done? 

        Examples
        --------
        >>> p = ProgressBar(1000)
        >>> for i in range(1001):
                p.animate(i)

        """
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            e_s = "No progress bar implementation without IPython" 
            raise NotImplementedError(e_s)

    def animate_ipython(self, iter, f_name=None):
        try:
            clear_output()
        except Exception:
            # terminal IPython has no clear_output
            pass
        print '\r', f_name, self, 
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete '%(elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) / 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


def probability_hist(data, bins=100, fig=None, cumsum=False, **kwargs):
    """
    A histogram, normalized as a probability density function. Note that this
    does not sum to 1. It's a probability *density* function, not a probability
    *mass* function.

    If cumsum is set to True, we get the cumulative probabilities

    """

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    # The denisty argument makes sure that it *integrates* to 1 over its
    # support:
    hist = np.histogram(data, bins=bins, density=True)
    if cumsum:
        ax.plot(hist[1][1:], np.cumsum(hist[0])/np.sum(hist[0]), **kwargs)
    else:
        ax.plot(hist[1][1:], hist[0], **kwargs)

    return fig


def sph2latlon(theta, phi):
    """Convert spherical coordinates to latitude and longitude.

    Returns
    -------
    lat, lon : ndarray
        Latitude and longitude.

    """
    return np.rad2deg(theta - np.pi/2), np.rad2deg(phi - np.pi)


def sig_on_projection(bvecs, val, ax=None, vmin=None, vmax=None,
                      cmap=matplotlib.cm.hot, cbar=True, tri=False,
                      boundary=True, **basemap_args):
    
    """Draw a signal on a 2D projection of the sphere.

    Parameters
    ----------

    r : (M, N) ndarray
        Function values.

    theta : (M,) ndarray
        Inclination / polar angles of function values.

    phi : (N,) ndarray
        Azimuth angles of function values.

    ax : mpl axis, optional
        If specified, draw onto this existing axis instead.

    cmap: mpl colormap

    cbar: Whether to add the color-bar to the figure
    
    basemap_args : dict
        Parameters used to initialise the basemap, e.g. ``projection='ortho'``.

    Returns
    -------
    m : basemap
        The newly created matplotlib basemap.

    """
    if ax is None: 
        fig, ax = plt.subplots(1)

    basemap_args.setdefault('projection', 'ortho')
    basemap_args.setdefault('lat_0', 0)
    basemap_args.setdefault('lon_0', 0)
    basemap_args.setdefault('resolution', 'c')

    from mpl_toolkits.basemap import Basemap

    m = Basemap(**basemap_args)
    if boundary:
        m.drawmapboundary()
    
    # Rotate the coordinate system so that you are looking from the north pole:
    bvecs_rot = np.array(np.dot(np.matrix([[0,0,-1],[0,1,0],[1,0,0]]), bvecs))


    # To get the orthographic projection, when the first coordinate is positive:
    neg_idx = np.where(bvecs_rot[0]>0)

    # rotate the entire bvector around to point in the other direction:
    bvecs_rot[:, neg_idx] *= -1
    
    _, theta, phi = geo.cart2sphere(bvecs_rot[0], bvecs_rot[1], bvecs_rot[2])
    lat, lon = sph2latlon(theta, phi)
    x, y = m(lon, lat)

    my_min = np.nanmin(val)
    if vmin is not None:
        my_min = vmin
        
    my_max = np.nanmax(val)
    if vmax is not None:
        my_max = vmax

    if tri:
        m.pcolor(x, y, val, vmin=my_min, vmax=my_max, tri=True, cmap=cmap)
    else:
        cmap_data = cmap._segmentdata
        red_interp, blue_interp, green_interp = (
        interp.interp1d(np.array(cmap_data[gun])[:,0],
                        np.array(cmap_data[gun])[:,1]) for gun in
                                                  ['red', 'blue','green'])
        
        r = (val - my_min)/float(my_max-my_min)

        # Enforce the maximum and minumum boundaries, if there are values
        # outside those boundaries:
        r[r<0]=0
        r[r>1]=1
        
        for this_x, this_y, this_r in zip(x,y,r):
            red = red_interp(this_r)
            blue = blue_interp(this_r)
            green = green_interp(this_r)
            m.plot(this_x, this_y, 'o',
                   c=[red.item(), green.item(), blue.item()])

    if cbar: 
        mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
        mappable.set_array([my_min, my_max])
        # setup colorbar axes instance.
        pos = ax.get_position()
        l, b, w, h = pos.bounds
        # setup colorbar axes
        cax = fig.add_axes([l+w+0.075, b, 0.05, h], frameon=False) 
        fig.colorbar(mappable, cax=cax) # draw colorbar

    return m, fig


def sphere(n=100):
    """

    Create a (grid) equi-sampled unit sphere with n samples

    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    return x,y,z



def plot_ellipsoid_mpl(Tensor, n=60):
    """
    Plot an ellipsoid from a tensor using matplotlib

    Parameters
    ----------

    Tensor: an ozt.Tensor class instance

    n: optional. If an integer is provided, we will plot this for a sphere with
    n (grid) equi-sampled points. Otherwise, we will plot the originally
    provided Tensor's bvecs.

    
    """

    Q = Tensor.Q

    x,y,z = sphere(n=n)
    new_bvecs = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    Tensor = ozt.Tensor(Tensor.Q, new_bvecs,
                        np.ones(new_bvecs.shape[-1]) * Tensor.bvals[0])

    v = Tensor.diffusion_distance.reshape(x.shape)
    
    r, phi, theta = geo.cart2sphere(x,y,z)
    x_plot, y_plot, z_plot = geo.sphere2cart(v, phi, theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x_plot, y_plot, z_plot,  rstride=2, cstride=2, shade=True)

    return fig


def plot_tensor_3d(Tensor, cmap='jet', mode='ADC', file_name=None,
                        colorbar=False, figure=None, vmin=None, vmax=None):

    """

    mode: either "ADC", "ellipse" or "pred_sig"
    """

    if not have_maya:
        e_s = "You can't use this function, unless you have mayavi installed" 
        raise ValueError(e_s)

    from mayavi.api import Engine
    # Tweak it: 
    engine = Engine()
    engine.start()

    
    Q = Tensor.Q
    sphere = create_unit_sphere(5)
    vertices = sphere.vertices
    faces = sphere.faces
    x,y,z = vertices.T 

    new_bvecs = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    Tensor = ozt.Tensor(Q, new_bvecs,
                        Tensor.bvals[0] * np.ones(new_bvecs.shape[-1]))

    if mode == 'ADC':
        v = Tensor.ADC
    elif mode == 'ellipse':
        v = Tensor.diffusion_distance
    elif mode == 'pred_sig':
        v = Tensor.predicted_signal(1)

    r, phi, theta = geo.cart2sphere(x,y,z)
    x_plot, y_plot, z_plot = geo.sphere2cart(v, phi, theta)

    tm = maya.triangular_mesh(x_plot, y_plot, z_plot, faces, scalars=v,
                         colormap=cmap)
    if colorbar:
        maya.colorbar(tm, orientation='vertical')

    #maya.xlabel('')
    #maya.ylabel('')
    #maya.zlabel('')

    scene = engine.scenes[0]
    scene.scene.background = (0.7529411764705882,
                              0.7529411764705882,
                              0.7529411764705882)

    # Set it to be aligned along the positive dimension of the y axis (similar
    # to the ortho projection plots). 
    scene.scene.y_plus_view()

    # Take care of the color-map:
    if vmin is None:
        vmin = np.min(v)
    if vmax is None:
        vmax = np.max(v)
        
    module_manager = engine.scenes[0].children[0].children[0].children[0]
    module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])
    module_manager.scalar_lut_manager.number_of_labels = 6

    # Take care of the lighting:
    scene.scene.light_manager.light_mode = 'vtk'

    
    scene.scene.render()
    
    if file_name is not None:
        scene.scene.save(file_name)

    engine.stop()
    return scene

def plot_signal_interp(signal, bvecs, cmap='jet', file_name=None,
                        colorbar=False, figure=None, vmin=None, vmax=None):

    """

    Interpolate a measured signal, using RBF interpolation.

    Parameters
    ----------
    signal:

    bvecs: the x,y,z locations where the signal was measured 
    
    """

    if not have_maya:
        e_s = "You can't use this function, unless you have mayavi installed" 
        raise ValueError(e_s)

    from mayavi.api import Engine
    # Tweak it: 
    engine = Engine()
    engine.start()

    
    s0 = Sphere(xyz=bvecs.T)
    s1 = create_unit_sphere(6)

    interp_signal = interp_rbf(signal, s0, s1)

    vertices = s1.vertices
    faces = s1.faces
    x,y,z = vertices.T 
    
    r, phi, theta = geo.cart2sphere(x,y,z)
    x_plot, y_plot, z_plot = geo.sphere2cart(interp_signal, phi, theta)

    tm = maya.triangular_mesh(x_plot, y_plot, z_plot, faces,
                              scalars=interp_signal,
                              colormap=cmap)
    if colorbar:
        maya.colorbar(tm, orientation='vertical')

    #maya.xlabel('')
    #maya.ylabel('')
    #maya.zlabel('')

    scene = engine.scenes[0]
    scene.scene.background = (0.7529411764705882,
                              0.7529411764705882,
                              0.7529411764705882)

    # Set it to be aligned along the positive dimension of the y axis (similar
    # to the ortho projection plots). 
    scene.scene.y_plus_view()

    # Take care of the color-map:
    if vmin is None:
        vmin = np.min(v)
    if vmax is None:
        vmax = np.max(v)
        
    module_manager = engine.scenes[0].children[0].children[0].children[0]
    module_manager.scalar_lut_manager.data_range = np.array([vmin, vmax])
    module_manager.scalar_lut_manager.number_of_labels = 6

    # Take care of the lighting:
    scene.scene.light_manager.light_mode = 'vtk'

    
    scene.scene.render()
    
    if file_name is not None:
        scene.scene.save(file_name)

    engine.stop()
    return scene
