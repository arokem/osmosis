import numpy as np



class Tensor(object):
    """
    Represent a diffusion tensor.
    """

    def __init__(self, Q):
        """
        Initialize a Tensor object
        """

        # Check the input:
        if Q.squeeze().shape==(9):
            Q = np.matrix(np.reshape(Q, (3,3)))

        elif Q.shape == (3,3):
            Q = np.matrix(Q)

        else:
            e_s = "Q had shape: ("
            e_s += ''.join(["%s, "%n for n in Q.shape])
            e_s += "), but should have shape (9) or (3,3)"
            raise ValueError(e_s)
    


