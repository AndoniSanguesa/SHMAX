from scipy.io import wavfile
from lyon.calc import LyonCalc

import os
from util import *

calc = LyonCalc()

### NOTE: lyon library only works on Linux!!! ###

def generate_cochs_for_all_files(path, train, region, speaker, decimation_factor=64):
    """
    Iterates through all files in the directory specified and generates cochleograms and
    extracts relevant metadata

    :param path: (String) path to the desired directory
    :param train: (bool) True if data belongs to training set, False otherwise
    :param region: (int) Region speaker belongs to
    :param speaker: (String) speaker identifier
    :param decimation_factor: (int) how much to decimate the model output
    :return: None
    """

    data_points = set()
    file_list = os.listdir(path)
    for file in file_list:
        data_points.add(file.split(".")[0])

    for datum in data_points:
        text = None
        words = None
        phone = None
        coch = None
        wav_path = None
        duration = None
        waveform = None
        relevant_files = filter(lambda x: datum in x, file_list)
        for file in relevant_files:
            full_path = path+file
            if full_path.endswith("TXT"):
                text = open(full_path).readlines()
            elif full_path.endswith("wav"):
                coch = create_cochleogram(full_path, decimation_factor)
                duration = get_wav_duration(full_path)
                wav_path = full_path
                _, waveform = wavfile.read(full_path)
            elif full_path.endswith("WRD"):
                words = open(full_path).readlines()
            elif full_path.endswith("PHN"):
                phone = open(full_path).readlines()
        save_coch_to_file(coch, duration, wav_path, train, region, speaker, phone, words, text, datum, waveform)



def generate_cochs_for_all_data_in_dir(path, decimation_factor=64):
    """
    Recursively generates cochleograms for all wav files in the specified directory or its
    subdirectories. (Makes python readable data)

    :param path: (String) path to desired directory
    :param decimation_factor: (int) how much to decimate the model output
    :return: None
    """

    for train_test_file in os.listdir(path):
        train_test_path = "./timit/data/"+train_test_file+"/"
        train = train_test_file == "TRAIN"
        for region_file in os.listdir(train_test_path):
            region_path = train_test_path + region_file+"/"
            region = int(region_file[-1])
            for speaker_file in os.listdir(region_path):
                speaker_path = region_path + speaker_file+"/"
                generate_cochs_for_all_files(speaker_path, train, region, speaker_file, decimation_factor)


def create_cochleogram(path, decimation_factor=64):
    """
    Generates cochleogram from wav file.

    :param path: (String) Path to wav file to generate cochleogram for
    :param decimation_factor: (int) how much to decimate the model output
    :return: ndarray of shape [N / decimation_factor, channels]
    """
    sample_rate, waveform = wavfile.read(path)
    waveform = waveform.astype("float64")
    coch = calc.lyon_passive_ear(waveform, sample_rate, decimation_factor)
    return coch

