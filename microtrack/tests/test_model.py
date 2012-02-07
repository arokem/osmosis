import microtrack.model as mtm

# Initially, we want to check whether the data is available (would have to be
# downloaded separately, because it's huge): 
data_path = os.path.split(mt.__file__)[0] + '/data/'
if 'dwi.nii.gz' in os.listdir(data_path):
    no_data = False
else:
    no_data = True


def test_Model():
    """

    Test the initialization of Model class objects
    
    """ 
