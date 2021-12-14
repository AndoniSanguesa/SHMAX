from spams import trainDL
from util import *
from cupyx.scipy.signal import convolve
import cupy
import random
import time


def SHMAX_2Ds(skip_sampling, skip_training, skip_inference, num_base, base_size, sample_size, stride, data_path, result_path):
    """
    Performs computations for SHMAX S layer for 2D data. Learns bases and performs inference.

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

    print("*** Learning Bases for Sparse Coding Layer ***")
    if skip_training:
        base = np.load(f"{result_path}/base_{num_base}_{sample_size}_{base_size}.npy")
    else:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        if skip_sampling:
            sample = np.load(f"{result_path}/sample_{num_base}_{sample_size}_{base_size}.npy")
        else:
            sample = sample_images_2d(sample_size, base_size, data_path)
            np.save(f"{result_path}/sample_{num_base}_{sample_size}_{base_size}.npy", sample)
        base = trainDL(sample, K=int(num_base),lambda1=1e0,iter=5e3,mode=1)

        np.save(f"{result_path}/base_{num_base}_{sample_size}_{base_size}.npy", base)

    print("*** Calculating Responses ***")
    if not skip_inference:
        random.seed(0)
        files = os.listdir(data_path)
        random.shuffle(files)
        for i in range(100):
            print(f"{round((i / len(files)) * 100, 3)}%")
            file = files[i]
            coch_info = load_coch_from_file(data_path+"/"+file)
            features = cupy.array(generate_features(coch_info["waveform"], sr=16000))
            w = np.zeros((features.shape[0]-base_size+1, features.shape[1]-base_size+1, int(num_base)))
            for j in range(int(num_base)):
                start = time.perf_counter()
                kernel = cupy.array(np.reshape(base[:, j], (int(base_size), int(base_size))))
                w[:,:,j] = convolve(features, kernel, mode="valid", method="direct").get()
                end = time.perf_counter()
                print(end - start)
            w = w[::stride, ::stride, :]
