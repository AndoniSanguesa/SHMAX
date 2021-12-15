from getPhonemeResponse import get_phoneme_times, ORDERED_PHONEMES
import os
import numpy as np
from scipy.io import savemat, loadmat
from dtw import dtw


def convert_category_list_to_dict(categories):
    """
    Converts phoneme categories represented by 2D list to a corresponding dictionary.

    :param categories: (List) List of lists where the sublists describe categories
    :return:(Dictionary) Maps phonemes to categories. Categories should be numerical and index from 0
    """

    ret_dict = {}

    for i, cat in enumerate(categories):
        for phone in cat:
            ret_dict[phone] = i

    return ret_dict


def generate_phoneme_matrices(corpus_data_path, phoneme_feature_path, result_path):
    """
    Saves a .mat file for every phoneme pronounced  in the timit corpus

    :param corpus_data_path: (String) Path to the corpus data
    :param phoneme_feature_path: (String) path to phoneme features
    :param result_path: (String) path to save phoneme files to
    :return: None
    """

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    phoneme_counter = [0]*len(ORDERED_PHONEMES)
    cnt = 0

    for file in os.listdir(phoneme_feature_path):
        print(cnt)
        phone_times = get_phoneme_times(corpus_data_path, file[:-4])

        data = loadmat(f"{phoneme_feature_path}/{file}")["data"]
        for phone, time in phone_times:
            phone_ind = ORDERED_PHONEMES.index(phone)
            new_mat = data[:, time[0]:time[1]]
            file_name = f"{phone}_{phoneme_counter[phone_ind]}.mat"
            savemat(f"{result_path}/{file_name}", {"data": new_mat})
            phoneme_counter[phone_ind] += 1
        cnt += 1


def abx_distance(stim1, stim2, ):
    """
    Calculates distance between two stimuli. Accounts for differences in stimuli duration using dynamic time warping

    :param stim1: (ndarray) The first stimulus. Should be shorter or equal to the length of the second stimulus
    :param stim2: (ndarray) The second stimulus.
    :param method: (String) The function used for omega (for comparing elements)
                    "arccos" --> arc cosine of the normalized dot product
                    "skl"    --> Symmetrised KL-Divergence
    :return: (float) the distance between the two stimuli for use in ABX testing
    """


def abx_testing(phoneme_mat_path, categories, num_categories, result_path=None, method="cosine", iter=100):
    """
    Performs machine ABX discrimination for all catagories against each other category. Creates confusion matrix based
    on results.

    :param phoneme_mat_path: (String) Path to phoneme matrices
    :param categories: (Dictionary) Maps phonemes to categories. Categories should be numerical and index from 0
    :param num_categories: (int) Number of categories represented by `categories`
    :param result_path: (String) Saves confusion matrix as .mat file to the desired path. If None, no matrix is saved
    :param method: (String) The function used for omega (for comparing elements)
                    "cosine"        --> arc cosine of the normalized dot product
                    "jensenshannon" --> Symmetrised KL-Divergence
    :param iter: (int) Number of iterations to compute ABX scores for
    :return: (ndarray) confusion matrix for ABX testing.
    """

    conf_mat = np.zeros((num_categories, num_categories))
    category_pops = [[] for _ in range(num_categories)]

    for file in os.listdir(phoneme_mat_path):
        phone = file.split("_")[0]
        category_pops[categories[phone]].append(file)
    for i in range(num_categories):
        for ind in range(iter):
            a = loadmat(f"{phoneme_mat_path}/{np.random.choice(category_pops[i])}")['data'].T
            x = loadmat(f"{phoneme_mat_path}/{np.random.choice(category_pops[i])}")['data'].T
            while x.shape == a.shape and (x == a).all():
                x = loadmat(f"{phoneme_mat_path}/{np.random.choice(category_pops[i])}")['data'].T
            dist_from_a = dtw(a, x, method).normalizedDistance
            for j in range(num_categories):
                if j == i:
                    continue
                for ind2 in range(iter):
                    b = loadmat(f"{phoneme_mat_path}/{np.random.choice(category_pops[j])}")['data'].T

                    dist_from_b = dtw(b, x, method).normalizedDistance
                    if dist_from_a < dist_from_b:
                        conf_mat[i, i] += 1
                    else:
                        conf_mat[i, j] += 1

    conf_mat = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]
    if result_path:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        savemat(result_path, {"data": conf_mat})

    return conf_mat

def general_classification_abx_testing(data_path, categories, num_categories, result_path=None, method="cosine", iter=100):
    """
    Performs machine ABX discrimination for all catagories against each other category. Creates confusion matrix based
    on results.

    :param data_path: (String) Path to phoneme matrices
    :param categories: (Dictionary) Maps phonemes to categories. Categories should be numerical and index from 0
    :param num_categories: (int) Number of categories represented by `categories`
    :param result_path: (String) Saves confusion matrix as .mat file to the desired path. If None, no matrix is saved
    :param method: (String) The function used for omega (for comparing elements)
                    "cosine"        --> arc cosine of the normalized dot product
                    "jensenshannon" --> Symmetrised KL-Divergence
    :param iter: (int) Number of iterations to compute ABX scores for
    :return: (ndarray) confusion matrix for ABX testing.
    """

    conf_mat = np.zeros((num_categories, num_categories))
    category_pops = [[] for _ in range(num_categories)]

    for file in os.listdir(data_path):
        phone = file.split("_")[0]
        category_pops[categories[phone]].append(file)
    for i in range(num_categories):
        for ind in range(iter):
            a = loadmat(f"{data_path}/{np.random.choice(category_pops[i])}")['data'].T
            x = loadmat(f"{data_path}/{np.random.choice(category_pops[i])}")['data'].T
            while x.shape == a.shape and (x == a).all():
                x = loadmat(f"{data_path}/{np.random.choice(category_pops[i])}")['data'].T
            dist_from_a = dtw(a, x, method).normalizedDistance
            distances = []
            for j in range(num_categories):
                if j == i:
                    distances.append(dist_from_a)
                    continue
                b = loadmat(f"{data_path}/{np.random.choice(category_pops[j])}")['data'].T

                dist_from_b = dtw(b, x, method).normalizedDistance
                distances.append(dist_from_b)
            classified_ind =  np.argmin(distances)
            conf_mat[i, classified_ind] += 1

    conf_mat = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]

    if result_path:
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        savemat(result_path, {"data": conf_mat})

    return conf_mat
