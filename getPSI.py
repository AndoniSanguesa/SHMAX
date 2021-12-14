import numpy as np
from scipy.stats import ranksums
import pickle

def pre_process(responses):
    m = max(responses) + 1
    n = max(responses[0]) + 1

    min_len = 0

    for i in range(m):
        for j in range(n):
            min_len = min(min_len, len(responses[i][j]))

    for i in range(m):
        for j in range(n):
            responses[i][j] = responses[i][j][:min_len-1]

    return responses

def get_psi(responses, save_path=None):
    """
    Gets PSI's given response to phonemes for specific units.

    :param responses: (ndarray) m x n representing unit activations to phonemes. m = # of phonemes, n = # of units
    :param save_path: (String) Path to save output to. Output will not be saved if None

    :return: PSI matrix of size m x n where PSI(i, j) represents the selectivity towards phone j for unit i
    """

    responses = pre_process(responses)

    m = max(responses)+1
    n = max(responses[0])+1

    psi = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            for k in range(m):
                resp1 = responses[i][j]
                resp2 = responses[k][j]

                if not resp1 or not resp2 or k == i:
                    continue

                _, p = ranksums(resp1, resp2, "greater")
                psi[i, j] += 1 if 0.1 > p else 0
    
    if save_path:
        np.save(save_path, psi)

    return psi
