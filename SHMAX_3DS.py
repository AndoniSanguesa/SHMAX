from util import *
from spams import trainDL_Memory
import os
from cupyx.scipy.signal import convolve
import cupy
import time


def SHMAX_3Ds(skip_sampling, skip_training, skip_inference, num_base, base_size, sample_size, stride, data_path, result_path):
    """
    Performs computations for SHMAX S layer for 3D data. Learns bases and performs inference.

    :param skip_sampling: (bool) If True, skip sampling images
    :param skip_training: (bool) If True, skip learning bases
    :param skip_inference: (bool) If True, skip inference step
    :param num_base: (int) number of bases to learn
    :param base_size: (int) size of base in bases
    :param sample_size: (int) size of sample to train bases with
    :param stride: (int) stride for convolution
    :param data_path: (String) path to where data should be loaded
    :param result_path: (String) path to where results should be saved
    :return: None
    """

    print("*** Learning Bases ***")
    if skip_training:
        base = np.load(f"{result_path}/base_{num_base}_{sample_size}_{base_size}.npy")
    else:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if skip_sampling:
            sample = np.load(f"{result_path}/sample_{num_base}_{sample_size}_{base_size}.npy")
        else:
            sample = sample_images_3d(sample_size, base_size, data_path)
            np.save(f"{result_path}/sample_{num_base}_{sample_size}_{base_size}.npy", sample)
        base = trainDL_Memory(sample, K=num_base, lambda1=1e0, iter=5e3, mode=1)
        np.save(f"{result_path}/base_{num_base}_{sample_size}_{base_size}.npy", base)

    print("*** Calculated Responses ***")
    if not skip_inference:
        files = os.listdir(data_path)
        for i in range(21  , len(files)):
            print(f"{round((i / len(files)) * 100, 3)}%")
            file = files[i]
            data = cupy.array(np.load(data_path+"/"+file))
            w = np.zeros((int(data.shape[0]-base_size+1), int(data.shape[1]-base_size+1), int(num_base)))
            for j in range(int(num_base)):
                kernel = cupy.array(np.reshape(base[:, j], (int(base_size), int(base_size), data.shape[2])))
                w[:,:,j] = np.reshape(convolve(data, kernel, mode="valid", method="direct").get(), w.shape[:2])
            w = w[::stride, ::stride, :]
            np.save(f"{result_path}/w_{i}.npy", w)