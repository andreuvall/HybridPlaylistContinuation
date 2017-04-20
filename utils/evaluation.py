# Evaluation utils.

from __future__ import print_function
from __future__ import division

import numpy as np


def nonzeros(m, row):
    """ returns the non zeroes of a row in csr_matrix """
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]


def mask_training_targets(output, train_target, verbose=True):
    """
    Set output values to -1. for known targets. This is in-place.
    (make sure not to recommend already known items)

    Parameters
    ----------
    output: np array, shape (#queries, #answers), probability for each pair
    train_target: sparse csr_array, shape (#queries, #answers), known pairs
    verbose: bool, print info if True
    """

    if verbose:
        print('\tMasking predicted output on known targets...')

    for u, output_u in enumerate(output):
        output_u[train_target[u].indices] = -1.


def find_rank(output, target, verbose=True):
    """
    For each nnz in target find the rank of its output within its sorted row.

    Parameters
    ----------
    output: np array, shape (#queries, #answers)
        Probability for each pair.
    target: sparse csr_array, shape (#queries, #answers)
        Binary observed pairs.
    verbose: bool
        Print info if True.

    Returns
    -------
    rank: list, length target.nnz
        Rank achieved by each nnz in target.
    """

    if verbose:
        print('\tFinding target ranks...')

    rank = []

    for u, output_u in enumerate(output):
        output_idx_sort = np.argsort(output_u)

        for target_idx, _ in nonzeros(target, u):
            is_hit = np.in1d(output_idx_sort, target_idx, assume_unique=True)
            rank.append(len(is_hit) - int(np.where(is_hit)[0]))

    return rank


def compute_map(output, target, verbose=True):
    """
    For each row compute the average precision of target using output.

    Parameters
    ----------
    output: np array, shape (#queries, #answers)
        Probability for each pair.
    target: sparse csr_array, shape (#queries, #answers)
        Binary observed pairs.
    verbose: bool
        Print info if True.

    Returns
    -------
    avgp_list: list, length #queries
        Average-precision for each row.
    """

    if verbose:
        print('\tComputing average precision ...')

    avgp_list = []

    for u, output_u in enumerate(output):
        output_idx_sort = np.argsort(output_u)
        target_idx = target[u].indices
        if not np.any(target_idx):  # check if there's something to validate
            continue                # some targets masked in MF for in-matrix

        precision, old_recall,  = 0., 0.

        for K, _ in enumerate(output_idx_sort, start=1):
            is_hit = np.in1d(output_idx_sort[-K:], target_idx, assume_unique=True)
            recall = is_hit.sum() / len(target_idx)
            if recall > old_recall:
                old_recall = recall
                precision += is_hit.sum() / K
                if recall == 1.:
                    break

        avgp_list.append(precision / len(target_idx))

    return avgp_list


def compute_precision_recall(output, target, K_list, verbose=True):
    """
    For each row compute the precision and recall at K of target using output.

    Parameters
    ----------
    output: np array, shape (#queries, #answers)
        Probability for each pair.
    target: sparse csr_array, shape (#queries, #answers)
        Binary observed pairs.
    K_list: list
        Each element is a list length we want to test on.
    verbose: bool
        Print info if True.

    Returns
    -------
    precision: dict
        Index K, list of precision scores for each row.
    recall: dict
        Index K, list of recall scores for each row.
    """

    if verbose:
        print('\tComputing precision and recall...')

    precision = {K: [] for K in K_list}
    recall = {K: [] for K in K_list}

    for u, output_u in enumerate(output):
        output_idx_sort = np.argsort(output_u)
        target_idx = target[u].indices
        if not np.any(target_idx):  # check if there's something to validate
            continue                # some targets masked in MF for in-matrix

        for K in K_list:
            is_hit = np.in1d(output_idx_sort[-K:], target_idx, assume_unique=True)
            num_hits = is_hit.sum()
            precision[K].append(num_hits / K)
            recall[K].append(num_hits / len(target_idx))

    return precision, recall
