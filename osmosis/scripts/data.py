import os
import osmosis as oz

data_path = os.path.split(oz.__file__)[0] + '/data/'

def make_data_set(root):
    """
    Create the full paths to the data given a root
    """ 
    exts = ['.nii.gz', '.bvecs', '.bvals']
    dwi, bvecs, bvals = [data_path + root + ext for ext in exts]
    return dwi, bvecs, bvals 

data_files = {1000:[make_data_set(
    '0009_01_DWI_2mm150dir_2x_b1000_aligned_trilin'),
    make_data_set(
    '0011_01_DWI_2mm150dir_2x_b1000_aligned_trilin')],
    2000:[make_data_set(
    '0005_01_DTI_2mm_150dir_2x_b2000_aligned_trilin'),
    make_data_set(
    '0007_01_DTI_2mm_150dir_2x_b2000_aligned_trilin')],
    4000:[make_data_set(
    '0005_01_DWI_2mm150dir_2x_b4000_aligned_trilin'),
    make_data_set(
    '0007_01_DWI_2mm150dir_2x_b4000_aligned_trilin')]}

# These are based on the calculations in GetADandRD
diffusivities = {1000: [dict(AD=1.7139,  RD=0.3887),
                        dict(AD=1.6986, RD=0.3760)],
                2000: [dict(AD=1.4291, RD=0.3507),
                       dict(AD=1.4202, RD=0.3357)],
                4000: [dict(AD=0.8403, RD=0.2369),
                       dict(AD=0.8375, RD=0.2379)]}


brain_mask = data_path + 'brainMask.nii.gz'
