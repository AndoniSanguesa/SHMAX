import os
import numpy as np
from util import *

def SHMAX_C(skip_pooling, pool_size, stride, data_path, result_path):
    """
     Performs computations for SHMAX C layer for 2D data.

    :param skip_pooling: (bool) If true, pooling is skipped
    :param pool_size: (int) Pooling size
    :param stride: (int) Stride amount for convolution
    :param data_path: (String) Path describing where data for this layer is stored
    :param result_path: (String) Path to where results should be stored
    :return: None
    """

    print("*** Calculating Responses ***")
    if not skip_pooling:
        files = os.listdir(data_path)
        for i in range(len(files)):
            print(f"{round((i / len(files)) * 100, 3)}%")
            file = files[i]
            data = np.load(data_path+"/"+file)
            y = np.zeros((data.shape[0]-1, data.shape[1]-1, data.shape[2]))
            for j in range(data.shape[2]):
                y[:,:,j] = col2im(np.max(im2col(data[:,:,j], [pool_size, pool_size]), 1), [pool_size, pool_size], data[:,:,j].shape)
            y = y[::stride, ::stride, :]
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            np.save(f"{result_path}/y_{i}.npy")