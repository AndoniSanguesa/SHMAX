import numpy as np
import os
import matplotlib.pyplot as plt
import wave
import matplotlib
import contextlib
from pycochleagram.cochleagram import cochleagram
from scipy.io import savemat, wavfile
from skimage.util import view_as_windows as viewW
import random
import PIL


def plot_psi_but_bad(psi):
    """
    Plots PSI graph but done in a horrible horrible way. I could not get pyplot to properlly strech the image. 

    :param psi: PSI matrix of size m x n where PSI(i, j) represents the selectivity towards phone j for unit i
    :return: None
    """
    cell_height = 25
    cell_width = 5
    num_cats = psi.shape[0]
    psi_plot = PIL.Image.new("L", (psi.shape[1]*cell_width, psi.shape[0]*cell_height))

    for i in range(psi.shape[0]):
        for j in range(psi.shape[1]):
            shade = int((psi[i, j]/num_cats) * 255)
            psi_plot.paste(PIL.Image.new("L", (cell_width, cell_height), shade), (j*cell_width, i*cell_height))
    
    psi_plot.save("AnalysisResults/psi_plot.png")


def plot_psi(psi):
    """
    Plots PSI graph.

    :param psi: PSI matrix of size m x n where PSI(i, j) represents the selectivity towards phone j for unit i
    :return: None
    """
    plt.matshow(psi, cmap="binary")

    plt.show()


def col2im(image_blocks, block_size, image_size):
    """
    Rearranges matrix columns into blocks

    :param image_blocks: (array-like) image blocks
    :param block_size: (array-like) size of blocks
    :param image_size: (ndarray) size of image to fit blocks into
    :return:
    """

    m,n = block_size
    mm,nn = image_size
    return image_blocks.reshape(nn-n+1,mm-m+1).T


def im2col(image, block_size, stepsize=1):
    """
    Rearrange image blocks into columns

    :param image: (array-like) The image to be rearranged
    :param block_size: (array-like) Size of blocks as tuple
    :param stepsize: (int) Stepsize to emulate the Matlab `sliding`
    :return: (ndarray) rearranged image
    """

    return viewW(image, (block_size[0],block_size[1])).reshape(-1,block_size[0]*block_size[1]).T[:,::stepsize]


def sample_images_3d(sample_size, base_size, path):
    """
    Samples 3d images for patches used to train bases.

    :param sample_size: (int) desired size of sample
    :param base_size: (int) desired size of base
    :param path: (Stirng) path to data files
    :return: (ndarray) samples as column vectors. Size: (base_size^2, sample_size)
    """

    print(f"*** Generating {sample_size} 3D samples from data ***")
    files = list(filter(lambda x: x.startswith("y"), os.listdir(path)))
    random.seed(0)
    random.shuffle(files)
    n = 100
    samples_per_image = int(sample_size // n)
    test_file = np.load(path+"/"+files[0])
    X = np.zeros((base_size**2 * test_file.shape[2], int(sample_size)))

    for i in range(n):
        file = files[i]
        data = np.load(path+"/"+file)
        x_vals = np.floor(np.random.rand(samples_per_image) * (data.shape[0] - int(base_size))).astype(int)
        y_vals = np.floor(np.random.rand(samples_per_image) * (data.shape[1] - int(base_size))).astype(int)
        for j in range(samples_per_image):
            patch = data[x_vals[j]:x_vals[j]+base_size, y_vals[j]:y_vals[j]+base_size, :]
            X[:,i*samples_per_image+j] = np.reshape(patch, ((base_size**2)*data.shape[2]))

    return np.asfortranarray(X, dtype="double")


def sample_images_2d(sample_size, base_size, path):
    """
    Samples 2d images for patches used to train bases.

    :param sample_size: (int) desired size of sample
    :param base_size: (int) desired size of base
    :param path: (Stirng) path to data files
    :return: (ndarray) samples as column vectors. Size: (base_size^2, sample_size)
    """

    print(f"*** Generating {sample_size} 2D samples from data ***")
    random.seed(0)
    files = os.listdir(path)
    random.shuffle(files)
    n = 100
    samples_per_image = int(sample_size // n)
    X = np.zeros((base_size**2, int(sample_size)))

    for i in range(n):
        file = files[i]
        coch_info = np.load(path+"/"+file)
        features = generate_features(coch_info["waveform"], sr=16000)
        x_vals = np.floor(np.random.rand(samples_per_image) * (features.shape[0] - base_size)).astype(int)
        y_vals = np.floor(np.random.rand(samples_per_image) * (features.shape[1] - base_size)).astype(int)
        for j in range(samples_per_image):
            patch = features[x_vals[j]:x_vals[j]+base_size, y_vals[j]:y_vals[j]+base_size]
            X[:,i*samples_per_image+j] = np.reshape(patch, (base_size**2))

    return np.asfortranarray(X, dtype="double")


def get_wav_duration(path):
    """
    Returns duration of specified wav file in seconds

    :param path: Path to wav file
    :return: (float) duration of wav file in seconds
    """

    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def generate_features(waveform, n_features=194, sr=1):
    """
    Generates a feature vector from cochleogram subbands given a waveform.

    :param waveform: (ndarray) array represnting waveform
    :param n_features: (int) number of features/subbands desired
    :param sr: (int) sampling rate that original wav was recorded in
    :return: (ndarray) list of subbands representing features
    """
    return cochleagram(waveform, sr, n_features, 0, 7630, 1, strict=False, ret_mode="subband", no_hp_lp_filts=True)


def generate_mat_for_all_data_in_dir(data_path, result_path, file_type="mat"):
    """
    Generates features and saves them as .mat files to desired path for each wav file in the path or subpaths.
    (Makes matlab readable data)

    :param data_path: Path to collect wav files from
    :param result_path: Path to save .mat files to
    :return: None
    """

    cnt = 0

    for train_test_file in os.listdir(data_path):
        train_test_path = data_path + train_test_file + "/"
        train_name = "train" if train_test_file == "TRAIN" else "test"
        for region_file in os.listdir(train_test_path):
            region_path = train_test_path + region_file + "/"
            region_name = f"{train_name}_{region_file[-1]}"
            for speaker_file in os.listdir(region_path):
                speaker_path = region_path + speaker_file + "/"
                speaker_name = f"{region_name}_{speaker_file}"
                for file in os.listdir(speaker_path):
                    if file.endswith(".wav"):
                        cnt += 1
                        file_name = f"{speaker_name}_{file[:-8]}.mat"
                        file_path = speaker_path + file
                        save_features_as_mat(file_path, f"{result_path}/{file_name}", sr=16000, file_type=file_type)


def save_features_as_mat(wav_path, output_path=None, n_features=194, sr=16000, file_type="mat"):
    """
    Generates feature vector from cochleogram subbands and saves them to a provided path as a .mat file

    :param wav_path: (String) path to wav file
    :param file_path: (String) path to save .mat file to
    :param n_features: (int) number of features/subbands desired
    :param sr: (int) sampling rate that original wav was recorded in
    :return: None
    """

    _, waveform = wavfile.read(wav_path)
    to_mat = generate_features(waveform, n_features, sr)
    if output_path:
        if file_type == "mat":
            savemat(output_path, {"data": to_mat})
        elif file_type == "npy":
            np.save(output_path, to_mat)
    return to_mat


def save_coch_to_file(coch, wav_len=None, wav_path=None, train=None, region=None, speaker=None,
                      phone=None, words=None, text=None, prompt=None, waveform=None, name=None):
    """
    Saves cochleogram to npz file for later use. Could be used to generate file in WSL and plotted on Windows for
    instance.

    :param coch: (ndarray) the cochleogram representation generated by lyon
    :param wav_len: (float) length of wav file in seconds
    :param wav_path: (String) Length of wav recording
    :param train: (bool) True if data belongs to training set, False otherwise
    :param region: (int) Region for speaker
    :param speaker: (String) Speaker identifier
    :param phone: (String) Phoneme transcription with time bounds for each phoneme
    :param words: (String) Text being read with time bounds for each word
    :param text: (String) full text being read, bounds only provided for full prompt
    :param prompt: (String) Prompt identifier
    :param waveform: (ndarray) array representation of wav file
    :param name: (String) Name of file to sve to. Defaults to metadata info.
    :return: None
    """
    path = ""
    if train is not None:
        path = "./pydata/" + ("TRAIN/" if train else "TEST/")
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if not name:
        name = f"{'train' if train else 'test'}-{region}-{speaker}-{prompt}"

    path += name

    np.savez(path,
             coch=coch,
             wav_len=wav_len,
             wav_path=wav_path,
             train=train,
             region=region,
             speaker=speaker,
             phone=phone,
             words=words,
             text=text,
             waveform=waveform)


def plot_wav(wav_path):
    """
    Plots a wav file
    
    :param wav_path: path to desired wav file 
    :return: None
    """

    sampleRate, audioBuffer = wavfile.read(wav_path)

    duration = len(audioBuffer) / sampleRate

    time = np.arange(0, duration, 1 / sampleRate)  # time vector

    plt.plot(time, audioBuffer)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

def load_coch_from_file(path):
    """
    Loads cochleogram and sample rate(if specified) from file.

    :param path: (String) path to file npz file cochleogram and sample rate was saved to
    :return: (cochleogram, wav_len) --> (ndarray of size [N / decimation_factor, channels], float)
    """

    return np.load(path)


def plot_coch(coch, wav_len=None, name=None):
    """
    Plots cochleogram using matplotlib's pyplot.

    :param coch: ndarray representation of cochleogram
    :param wav_len: (float) Length of wav file in seconds
    :param name: (String) name of wav file. Used for plot title
    :return: None
    """

    xs = []
    ys = []
    zs = []

    y_size, x_size = coch.shape

    for y_val in range(y_size):
        for x_val in range(0, x_size, 20):
            xs.append(x_val/x_size * wav_len if wav_len else x_val)
            ys.append(y_val)
            zs.append(coch[y_val][x_val])

    perm = sorted(list(range(len(xs))), key=lambda k: -ys[k])
    new_xs = [xs[i] for i in perm]
    new_ys = [ys[i] for i in perm]
    new_zs = [zs[i] for i in perm]

    plt.scatter(new_xs, new_ys, c=new_zs, cmap="viridis", s=np.array(new_zs)*20)
    plt.ylabel("Frequency [mHz]")
    plt.xlabel("Time [sec]")
    if name:
        plt.title(f"{name}")
    else:
        plt.title(f"Cochleogram")
    plt.show()

