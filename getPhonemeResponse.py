import os
import numpy as np
import h5py
import pickle

NON_PHONE_CODES = ["pau", "epi", "h#", "1", "2"]
ORDERED_PHONEMES = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el',
                    'em', 'en', 'eng', 'er', 'ey', 'f', 'g', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n',
                    'ng', 'nx', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z',
                    'zh']
PHONE_DICT = {key: i for i, key in enumerate(ORDERED_PHONEMES)}

def get_timit_phoneme_file(base_path, name):
    """
    Gets path to desired file assuming the Timit corpus and correct formatting for `name`

    :param base_path: (String) path to /timit/data/
    :param name: (String) Name of output file from SHMAX

            e.g. test_1_FAKS0_SI943

    :return: (String) path to desired file
    """
    conds = name.split("_")
    return f"{base_path}/{'TEST' if conds[0] == 'test' else 'TRAIN'}/DR{conds[1]}/{conds[2]}/{conds[3]}.PHN"


def get_phoneme_times(corpus_data_path, name):
    """
    Returns list of tuples with phoneme names and their start/end times. Will ignore pauses, beginning/ending markers,
    and stress markers.

    e.g. [('a', [1000, 1500]), ('b', [1500, 2321])...]

    NOTE: function may have to be modified to work with other corpora of course.

    :param corpus_data_path: (String) base path to corpus data.
    :param name: (String) name or portion of name for desired file. (Should not be ambiguous!)
    :return: (List) Returns list of tuples with phonemes and associated times
    """

    desired_file = get_timit_phoneme_file(corpus_data_path, name)
    ret_list = []
    corrected_start = -1
    with open(desired_file) as f:
        for line in f.readlines():
            start, end, phone = line.split()

            if phone.lower() in NON_PHONE_CODES:
                continue

            if "cl" in phone:
                corrected_start = start
                continue

            if corrected_start != -1:
                start = corrected_start
                corrected_start = -1

            ret_list.append((phone, [int(start), int(end)]))
    return ret_list


def get_timit_length(corpus_data_path, name):
    """
    Returns full length of timit data file. Used to scale phoneme times to SHMAX output

    :param corpus_data_path: (String) path to /timit/data/
    :param name: (String) Name of output file from SHMAX
    :return: (int) length of desired sample
    """

    desired_file = get_timit_phoneme_file(corpus_data_path, name)[:-4] + ".TXT"

    with open(desired_file) as f:
        return int(f.readline().split()[1])


def remap_times(phone_times, timit_len, data_len):
    """
    Length of output from SHMAX does not match length of timit data, so the phoneme times must be modified to account
    for this difference.

    :param phone_times: (list) Phone times as described by TIMIT dataset
    :param timit_len: (int) Length of sample as defined by TIMIT dataset
    :param data_len: (int) Length of SHMAX output
    :return: (list) Returns list of tuples with phonemes and associated times
    """

    mod_phn = lambda phn: (phn[0], list(map(lambda x: int(data_len*(x/timit_len)),phn[1])))

    return list(map(mod_phn, phone_times))


def get_phoneme_response(corpus_data_path, SHMAX_data_path, save_path=None, num_units=500, num_categories=52, method="max",
                         categories=None):
    """
    Gets the response to all phonemes for all units of the current SHMAX model output.

    :param corpus_data_path: (String) Path to corpus data
    :param SHMAX_data_path: (String) Path that SHMAX model output is saved to
    :param save_path: (String) Path to save responses to. Does not save if None.
    :param num_units: (int) Number of units present in SHMAX model
    :param num_categories: (int) Number of categories represented by the `categories` parameter
    :param method: (String) Method of determining unit activation
            given n values for a given unit and time, activation for that unit at that time is calculated as:
            "max" --> the max of the n values
            "mean" --> the average of the n values
    :param categories: (Dictionary) Maps phonemes to categories. Categories should be numerical and index from 0
    :return: (ndarray) m x n matrix describing unit activation for phonemes. m = # of phonemes or categories,
             n = # of units
    """

    if categories is None:
        categories = PHONE_DICT

    unit_response = {i:{j:[] for j in range(num_units)} for i in range(num_categories)}
    category_count = [0]*num_categories
    prog = 0

    for file in os.listdir(SHMAX_data_path):
        print(prog / len(os.listdir(SHMAX_data_path)))
        if "base" in file:
            continue

        data = h5py.File(f"{SHMAX_data_path}/{file}")["y"]
        file_len = get_timit_length(corpus_data_path, file[2:-4])
        raw_phone_times = get_phoneme_times(corpus_data_path, file[2:-4])
        phone_times = remap_times(raw_phone_times, file_len, data.shape[1])
        for pt in phone_times:
            phone_ind = categories[pt[0]]
            for unit in range(num_units):
                unit_response[phone_ind][unit].append(np.mean(data[unit, pt[1][0]:pt[1][1], 0]))
            category_count[phone_ind] += 1

    if save_path:
        pickle.dump(unit_response, open(save_path, "wb"))

    return unit_response
