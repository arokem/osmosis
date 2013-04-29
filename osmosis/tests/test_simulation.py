import numpy as np
import numpy.testing as npt
import osmosis.simulation as sims

def test_signal_1d():
    """
    Test simulations of 1D signals
    """
    # This is generic
    theta = np.linspace(0,2*np.pi,100)

    # One fiber:
    fiber_weights = 0.5
    d_para = 1.5
    d_ortho = 0.5
    phi = 0
    iso_weights = 0.5
    d_iso = 0.5

    sims.signal_1d(theta, 1, fiber_weights, d_para, d_ortho, phi, iso_weights,
                   d_iso)

    # Several fibers with same d_para, d_ortho
    fiber_weights = [0.1,0.2,0.1]
    d_para = 1.5
    d_ortho = 0.5
    phi = [0, np.pi/4, np.pi/2]
    iso_weights=0.6
    d_iso = 0.5

    sims.signal_1d(theta, 1, fiber_weights, d_para, d_ortho, phi, iso_weights,
                   d_iso)


    # Several fibers with different d_para, d_ortho
    fiber_weights = [0.1,0.2,0.1]
    d_para = [1.5, 1.4, 1.3]
    d_ortho = [0.5, 0.3, 0.8]
    phi = [0, np.pi/4, np.pi/2]
    iso_weights=0.6
    d_iso = 0.5

    sims.signal_1d(theta, 1, fiber_weights, d_para, d_ortho, phi, iso_weights,
                   d_iso)

    # One fiber, several isotropic components with one d_iso:
    fiber_weights = 0.1
    d_para =  1.3
    d_ortho = 0.5
    phi = 0
    iso_weights= [0.6, 0.2]
    d_iso = 0.5

    sims.signal_1d(theta, 1, fiber_weights, d_para, d_ortho, phi, iso_weights,
                   d_iso)
    
    # One fiber, several isotropic components with different d_iso:
    fiber_weights = 0.1
    d_para =  1.3
    d_ortho = 0.5
    phi = 0
    iso_weights= [0.6, 0.2]
    d_iso = [0.5, 0.6]

    sims.signal_1d(theta, 1, fiber_weights, d_para, d_ortho, phi, iso_weights,
                   d_iso)
