from scipy.stats import ttest_ind
import numpy as np


def get_active_units(responses, silence_index):
    """
    Gets the active units of SHMAX response

    :param responses: (ndarray) m x n x k array outputted from the SHMAX model
    :param silence_index: (int) Index of silent frame. Needs to be of size n
    :return: (ndarray) boolean index of length k that indicate active units
    """

    ret_vec = np.zeros(responses.shape[2])

    for i in range(responses.shape[2]):
        r = np.random.permutation(list(range(responses.shape[1])))
        x = np.zeros(responses.shape[:2])
        for j in range(responses.shape[1]):
            x[:,j] = responses[:,r[j], i]
        y = responses[:, silence_index, i]
        ret_vec[i] = 1 if ttest_ind(x, y)[1] < 1e-3 else 0

    return ret_vec