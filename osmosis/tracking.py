# Track using fODF models

import numpy as np
import dipy.reconst.interpolate as interp
import dipy.tracking.markov as dpt
import dipy.tracking.utils as dpu
import dipy.data as dpd

import osmosis.fibers as ozf

def track(model, data, sphere=None, step_size=1, angle_limit=20, seeds=None,
          density=[2,2,2], voxel_size=[1,1,1]):
    """
    Interface for tracking based on fiber ODF models

    `model` needs to have a `fit` method, such that model.fit(data).odf(sphere)
    is a legitimate ODF (that is has dimensions (x,y,z, n_vertices), where
    n_vertices refers to the vertices of the provided sphere. 
    
    """

    # If no sphere is provided, we will use the dipy symmetrical sphere with
    # 724 vertcies. That should be enough
    if sphere is None:
        sphere = dpd.get_sphere('symmetric724')

    stepper = dpt.FixedSizeStepper(step_size)
    interpolator = dpt.NearestNeighborInterpolator(data, voxel_size)
    
    if seeds is None:
        seeds = dpu.seeds_from_mask(mask, density, voxel_size)

    pwt = dpt.ProbabilisticOdfWeightedTracker(model, interpolator, mask,
                                              stepper, angle_limit, seeds,
                                              sphere)

    pwt_streamlines = list(pwt)

    fibers = []
    for f in pwt_streamlines:
          fibers.append(ozf.Fiber(f))
