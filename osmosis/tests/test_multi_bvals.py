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

all_b_idx_t = np.array([3,4,5,6,7])

bvecs_t = np.zeros([3,8])
bvecs_t[0,:] = np.array([1,0,0,-1,0,0,1/np.sqrt(3),-1/np.sqrt(3)])
bvecs_t[1,:] = np.array([0,1,0,0,-1,0,1/np.sqrt(3),-1/np.sqrt(3)])
bvecs_t[2,:] = np.array([0,0,1,0,0,-1,1/np.sqrt(3),-1/np.sqrt(3)])

# Mock data to be used in all tests
data_t = np.zeros([2,2,2,8])
#Data for b values 0.005, 0.005, 0.010
data_t[:,:,:,0:3] = 2000 + abs(np.random.randn(2,2,2,3)*500)
#Data for b value 2.010
data_t[:,:,:,3] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200))
#Data for b values 1.005, 0.950
data_t[:,:,:,4:6] = 1000 + abs(np.random.randn(2,2,2,2)*500)
#Data for b values 1.950
data_t[:,:,:,6] = np.squeeze(500 + abs(np.random.randn(2,2,2,1)*200))
#Data for b value 1.000
data_t[:,:,:,7] = np.squeeze(1000 + abs(np.random.randn(2,2,2,1)*500))

# Mock mask to be used in all tests
mask_t = np.zeros([2,2,2])
mask_t[:,:,1] = 1

# Rotational vectors = # vertices
rot_vecs_t = bvecs_t

# Initalize the sfm multi b values class
ad = {1000:1.5, 2000:1.5, 3000:1.5}
rd = {1000:0.5, 2000:0.5, 3000:0.5}
mb = sfm_mb.SparseDeconvolutionModelMultiB(data_t, bvecs_t, bvals_t, mask = mask_t,
                                           axial_diffusivity = ad,
                                           radial_diffusivity = rd,
                                           params_file = 'temp')

def test_response_function():
    rf_bvec = np.reshape(bvecs_t[:,4],(3,1))
    tensor_t = ozt.Tensor(np.diag([1.5, 0.5, 0.5]), rf_bvec, np.array([1000]))
    
    evals_a, evecs_a = mb.response_function(1000, rf_bvec).decompose
    evals_t, evecs_t = tensor_t.decompose
    
    npt.assert_equal(evals_t, evals_a)
    npt.assert_equal(evecs_t, evecs_a)
    
    return evals_t, evecs_t
    
def test_rotations():
    evals_t, evecs_t = test_response_function()
    out_t = np.empty((5, 1))

    for idx, bvec in enumerate(rot_vecs_t[:,3:8].T):
        this_rot_t = ozt.rotate_to_vector(bvec, evals_t, evecs_t,
               np.reshape(bvecs_t[:,4], (3,1)), np.array([bvals_scaled_t[4]/1000]))
        out_t[idx] = this_rot_t.predicted_signal(1)
        
    npt.assert_equal(out_t, mb._calc_rotations(bvals_scaled_t[4]/1000,
                                     np.reshape(bvecs_t[:,4], (3,1))))
    
    return out_t
    
def test_relative_signal():
    S0_t = np.mean(data_t[...,np.array([0,1,2])],-1)
    sig_t = data_t[...,np.array([4,5,7])]
    signal_rel_t = sig_t/np.reshape(S0_t, (2,2,2,1))

    npt.assert_equal(signal_rel_t, mb.relative_signal[..., bval_ind_rm0_t[0]])
    
    return signal_rel_t
    
def test__flat_relative_signal():
    out_flat = test_relative_signal()[np.where(mask_t)]
    npt.assert_equal(out_flat, mb._flat_relative_signal[...,bval_ind_rm0_t[0]])
        
def test_regressors():
    _, tensor_regressor_a, design_matrix_a, fit_to_a, fit_to_means_a = mb.regressors

    fit_to_t = np.empty((np.sum(mask_t), len(bvals_scaled_t[3:])))
    fit_to_means = np.empty((np.sum(mask_t), len(bval_ind_t[3:])))

    n_columns = len(bvals_scaled_t[np.where(bvals_scaled_t > 0)])
    tensor_regressor_t = np.empty((n_columns, n_columns))
    design_matrix_t = np.empty(tensor_regressor_t.shape)
    for idx, b_idx in enumerate(all_b_idx_t):
        this_fit_to = mb._flat_relative_signal[:, idx]
        fit_to_t[:, idx]  = this_fit_to - mb._flat_rel_sig_avg(bvals_t, b_idx)
            
        # Check tensor regressors
        this_tensor_regressor = np.squeeze(mb._calc_rotations(bvals_t[b_idx],
                                                np.reshape(bvecs_t[:, b_idx], (3,1))))
        tensor_regressor_t[idx] = this_tensor_regressor
        
        #Check design matrix
        this_MD = (1.5 + 2*0.5)/3
        design_matrix_t[idx] = this_tensor_regressor - np.exp(-bvals_t[b_idx]*this_MD)
            
    npt.assert_equal(tensor_regressor_t, tensor_regressor_a)
    npt.assert_equal(fit_to_t, fit_to_a)
    
    return tensor_regressor_t, design_matrix_t, fit_to_t
    
def test__n_vox():
    npt.assert_equal(4, mb._n_vox)

def test_design_matrix():
    _, tensor_regressor_a, design_matrix_a, _, _ = mb.regressors
    tensor_regressor_t, design_matrix_t, fit_to_t = test_regressors()

    npt.assert_equal(design_matrix_t, design_matrix_a)

    for rv in np.arange(len(mb.rot_vecs[:,3:8].T)):
        # Test to see if the mean across the vertices of the design matrix == 0
        mean_3rot = np.mean(design_matrix_a[:,rv])
        npt.assert_equal(mean_3rot<0.5,1)

def test_model_params():
    _, _, fit_to_t = test_regressors()
    _, _, design_matrix_a, _, _ = mb.regressors

    npt.assert_equal(mb._fit_it(fit_to_t[0], design_matrix_a), 
                        mb.model_params[mb.mask][0])
    
def test_fit():
    diff_means = np.mean(data_t[0,0,0,3:8]) - np.mean(mb.fit[np.where(mask_t)][0])
    npt.assert_equal(abs(diff_means)<400,1)
    
def test_predict():
    bvec_t = np.reshape(bvecs_t[:,3], (3,1))
    tensor_regressor_t = mb._calc_rotations(np.array([2]), bvec_t)

    this_MD = (1.5 + 2*0.5)/3
    design_matrix_t = tensor_regressor_t.T - np.exp(-bvals_t[3]*this_MD)

    fit_to_mean_t = np.zeros((4,1))
    out_t = np.zeros((4,1))

    fit_to_mean_t[:, 0] = mb._flat_rel_sig_avg(np.array([2]), 0)

    for vox in np.arange(4):
        this_params_t = mb._flat_params[vox]
        this_params_t[np.isnan(this_params_t)] = 0.0
        out_t[vox] = (np.dot(this_params_t, design_matrix_t.T) +
                            fit_to_mean_t[vox]) * mb._flat_S0[vox]
                            
        npt.assert_equal(abs(np.squeeze(out_t[vox]) - mb.predict(bvec_t,
                                           np.array([2]))[vox]) < 30, 1)