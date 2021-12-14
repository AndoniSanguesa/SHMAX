from sklearn.cluster import AgglomerativeClustering
import numpy as np


def cluster_psi(psi, save_path=None):
    """
    Performs hierarchical clustering on input data. Clusters phoneme rows with similar activations per unit and clusters
    units with similar activations per phoneme.

    :param psi: (ndarray) PSI matrix of size m x n where PSI(i, j) represents the selectivity towards phone j for unit i
    :param save_path: (String) Path to save output to. Output will not be saved if None
    :return: (ndarray) modified PSI matrix of size m x n after clustering
    """

    phoneme_clustering = AgglomerativeClustering(n_clusters=6).fit(psi).labels_
    unit_clustering = AgglomerativeClustering(n_clusters=6).fit(psi.T).labels_

    phoneme_permute = []
    unit_permute = []

    for clust in range(6):
        phoneme_permute.extend(np.where(phoneme_clustering == clust)[0].tolist())
        unit_permute.extend(np.where(unit_clustering == clust)[0].tolist())

    phoneme_idx = np.empty_like(phoneme_permute)
    unit_idx = np.empty_like(unit_permute)

    phoneme_idx[phoneme_permute] = np.arange(len(phoneme_permute))
    unit_idx[unit_permute] = np.arange(len(unit_permute))

    psi = psi[phoneme_idx, :]
    psi = psi[:, unit_idx]

    if save_path:
        np.save(save_path, psi)

    return psi