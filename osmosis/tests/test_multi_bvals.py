import numpy as np
import osmosis.tensor as ozt
import numpy.testing as npt

import osmosis.multi_bvals as sfm_mb

# Mock b value array to be used in all tests
bvals_t = np.array([0.005, 0.005, 0.010, 2.010, 1.005, 0.950, 1.950, 1.000])

b_idx = 1
bval_list_t = [(np.array([0, 0, 0]))]
bval_list_t.append(np.array([1000, 1000, 1000]))
bval_list_t.append(np.array([2000, 2000]))

bval_ind_t = [np.array([0,1,2]), np.array([4,5,7]), np.array([3,6])]
bval_ind_rm0_t = [np.array([1,2,4]), np.array([0,3])]

bvals_scaled_t = np.array([0, 0, 0, 2000, 1000, 1000, 2000, 1000])

unique_b_t = np.array([0,1,2])

bvecs_t = np.zeros([3,8])
bvecs_t[0,:] = np.array([1,0,0,-1,0,0,1/np.sqrt(3),-1/np.sqrt(3)])
bvecs_t[1,:] = np.array([0,1,0,0,-1,0,1/np.sqrt(3),-1/np.sqrt(3)])
bvecs_t[2,:] = np.array([0,0,1,0,0,-1,1/np.sqrt(3),-1/np.sqrt(3)])

# Mock data to be used in all tests
data_t = np.zeros([2,2,2,8])
data_t[:,:,:,0:3] = 2000 + abs(np.random.randn(2,2,2,3)*500) #Data for b values 0.005, 0.005, 0.010
data_t[:,:,:,3] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) #Data for b value 2.010
data_t[:,:,:,4:6] = 1000 + abs(np.random.randn(2,2,2,2)*500) #Data for b values 1.005, 0.950
data_t[:,:,:,6] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200)) #Data for b values 1.950
data_t[:,:,:,7] = np.squeeze(1000 + abs(np.random.randn(2,2,2,1)*500)) #Data for b value 1.000

# Mock mask to be used in all tests
mask_t = np.zeros([2,2,2])
mask_t[:,:,1] = 1

# Rotational vectors = # vertices
rot_vecs_t = bvecs_t

# Initalize the sfm multi b values class
mb = sfm_mb.SparseDeconvolutionModelMultiB(data_t, bvecs_t, bvals_t, mask = mask_t, params_file = 'temp')


def test_response_function():
    rf_bvecs = bvecs_t[:,np.array([4,5,7])]
    tensor_t = ozt.Tensor(np.diag([1, 0, 0]), rf_bvecs, np.array([1000, 1000, 1000]))
    
    evals_a, evecs_a = mb.response_function(0).decompose
    evals_t, evecs_t = tensor_t.decompose
    
    npt.assert_equal(evals_t, evals_a)
    npt.assert_equal(evecs_t, evecs_a)
    
    return evals_t, evecs_t
    
def test_rotations():
    evals_t, evecs_t = test_response_function()
    this_b_inds_t = np.array([4,5,7])
    out_t = np.empty((5, 3))

    for idx, bvec in enumerate(rot_vecs_t[:,3:8].T):
        this_rot_t = ozt.rotate_to_vector(bvec, evals_t, evecs_t, bvecs_t[:,this_b_inds_t], bvals_scaled_t[this_b_inds_t]/1000)
        out_t[idx] = this_rot_t.predicted_signal(1)
        
    npt.assert_equal(out_t, mb.rotations(0))
    
    return out_t
    
def test_relative_signal():
    S0_t = np.mean(data_t[...,np.array([0,1,2])],-1)
    sig_t = data_t[...,np.array([4,5,7])]
    signal_rel_t = sig_t/np.reshape(S0_t, (2,2,2,1))

    npt.assert_equal(signal_rel_t, mb.relative_signal[..., bval_ind_t[1]])
    
    return signal_rel_t
    
def test__flat_relative_signal():
    out_flat = test_relative_signal()[np.where(mask_t)]
    npt.assert_equal(out_flat, mb._flat_relative_signal[...,bval_ind_t[1]])
        
def test_regressors():
    tensor_regressor_list_t = [mb.rotations(0), mb.rotations(1)]
    _, tensor_regressor_list_a, fit_to_list_a = mb.regressors

    fit_to_list_t = list()
    for idx, b in enumerate(unique_b_t[1:]):
        fit_to_list_t.append(mb._flat_relative_signal[:,bval_ind_t[1:][idx]].T)
        npt.assert_equal(tensor_regressor_list_t[idx], tensor_regressor_list_a[idx])
        npt.assert_equal(fit_to_list_t[idx], fit_to_list_a[idx])
        
    return tensor_regressor_list_t, fit_to_list_t
    
def test__n_vox():
    npt.assert_equal(4, mb._n_vox)

def test_design_matrix():
    _, tensor_regressor_list_a, fit_to_list_a = mb.regressors
    
    for bi in np.arange(len(mb.b_inds_rm0)):
        evals_a, evecs_a = mb.response_function(bi).decompose
        this_tr = mb.design_matrix[mb.b_inds_rm0[bi]] + np.mean(tensor_regressor_list_a[bi], -1)
        for rv in np.arange(len(mb.rot_vecs[:,3:8].T)):
            # Check to see if contents of the rotational vectors are equal to the
            # predicted signal
            this_rot = ozt.rotate_to_vector(mb.rot_vecs[:,3:8].T[0], evals_a, evecs_a,
                     mb.bvecs[:,mb.b_inds[bi]], mb.rounded_bvals[mb.b_inds[bi]]/1000)
            npt.assert_equal(this_tr[:,0],this_rot.predicted_signal(1))
            
            # Test to see if the mean across the vertices of the design matrix == 0
            mean_3rot = np.mean(mb.design_matrix[bval_ind_rm0_t[bi]][:,rv])
            npt.assert_equal(mean_3rot<1*10**-10,1)

def test_model_params():
    _, fit_to_list_t = test_regressors()

    this_fit_to = np.empty(5)
    for bi in np.arange(len(bval_ind_rm0_t)):
        this_fit_to[bval_ind_rm0_t[bi]] = fit_to_list_t[bi].T[0] - np.mean(fit_to_list_t[bi].T[0])

    npt.assert_equal(mb._fit_it(this_fit_to, mb.design_matrix), mb.model_params[mb.mask][0])
    
def test_fit():
    diff_means = np.mean(data_t[0,0,0,3:8]) - np.mean(mb.fit[np.where(mask_t)][0])
    npt.assert_equal(abs(diff_means)<400,1)