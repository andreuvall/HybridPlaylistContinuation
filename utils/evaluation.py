# Evaluation utils.

from __future__ import print_function
from __future__ import division

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

import numpy as np

import copy


def mask_array_items(a, mask, verbose=True):
    """
    Set `a` items to -1 in the given `mask` positions. This is in-place.
    (Basically used to ensure that we do not give known answers to queries.)

    Parameters
    ----------
    a: np array, shape (n, m)
        Dense array.
    mask: sparse csr_matrix, shape (n, m)
        Sparse binary mask.
    verbose: bool
        Print info if True.
    """

    if verbose:
        print('\tMasking array items on given mask...')

    for u, mask_u in enumerate(mask):
        a[u, mask_u.indices] = -1.

    # # alternative code would be
    # mask_coo = mask.tocoo()
    # a[mask_coo.row, mask_coo.col] = -.1


def mask_array_rows(sp, mask, verbose=True):
    """
    Set to 0 the rows of `sp` given by the `mask` indices. This is in-place.
    (Basically used to ensure that we do not evaluate CF on unknown answers.)

    Parameters
    ----------
    sp: sparse csr_matrix, shape (n, m)
        Sparse binary array.
    mask: numpy array, shape (n, )
        Row indices that should be set to 0.
    verbose: bool
        Print info if True.
    """

    if verbose:
        print('\tMasking array rows on given mask...')

    # clear rows in a
    sp[mask, :] = 0
    sp.eliminate_zeros()


def mask_array_cols(sp, mask, verbose=True):
    """
    Set to 0 the cols of `sp` given by the `mask` indices. This is in-place.
    (Basically used to ensure that we do not evaluate CF on unknown answers.)

    Parameters
    ----------
    sp: sparse csr_matrix, shape (n, m)
        Sparse binary array.
    mask: numpy array, shape (m, )
        Col indices that should be set to 0.
    verbose: bool
        Print info if True.
    """

    if verbose:
        print('\tMasking array cols on given mask...')

    # clear cols in a
    sp[:, mask] = 0
    sp.eliminate_zeros()


def find_rank(scores, targets, verbose=True):
    """
    For each (playlist, song) pair in `targets`, find the rank of its `scores`
    within the full list of sorted `scores` for this playlist continuation.

    Parameters
    ----------
    scores: np array, shape (num_playlists, num_songs)
        Probability or score for each (playlist, song) pair.
    targets: sparse csr_matrix, shape (num_playlists, num_songs)
        Binary sparse array indicating target playlist continuations.
    verbose: bool
        Print info if True.

    Returns
    -------
    rank: list, length targets.nnz
        Rank achieved by each nnz in `targets`.
    """

    if verbose:
        print('\tFinding ranks...')

    rank = []

    # find rows where `targets` is not empty
    # (https://mike.place/2015/sparse/)
    non_empty_idx = np.where(np.diff(targets.indptr) != 0)[0]
    num_empty = targets.shape[0] - len(non_empty_idx)
    if num_empty > 0 and verbose:
        print('\t\tWarning: \'targets\' has {} empty rows. Is that ok?'.format(num_empty))

    for u in non_empty_idx:

        # sort all query output values (argsort sorts low to high)
        scores_u_idx_sort = np.argsort(scores[u])[::-1]

        for targets_ui in targets[u].indices:
            rank_ui = int(np.where(scores_u_idx_sort == targets_ui)[0])
            rank.append(rank_ui + 1)  # make the rank 1-based

    return rank


def compute_average_precision(scores, targets, verbose=True):
    """
    Compute the average precision of each target continuation in `targets`
    given the playlist-song `scores`. Precisely, we find the cut points where
    recall increases and compute the precision there. Averaging them gives the
    mean average precision. The precision at the first cut point is the mean
    reciprocal rank. References:
    - Chap. 4.3.2. Baeza-Yates and Ribeiro-Neto. Modern Information Retrieval
    - https://www.youtube.com/watch?v=pM6DJ0ZZee0

    Parameters
    ----------
    scores: np array, shape (num_playlists, num_songs)
        Probability or score for each (playlist, song) pair.
    targets: sparse csr_matrix, shape (num_playlists, num_songs)
        Binary sparse array indicating target playlist continuations.
    verbose: bool
        Print info if True.

    Returns
    -------
    rr: list, length num_playlists
        Reciprocal rank of each target continuation.
    avgp: list, length num_playlists
        Average-precision of each target continuation.
    """

    if verbose:
        print('\tComputing average precision...')

    rr = []
    avgp = []

    # find rows where target is not empty (https://mike.place/2015/sparse/)
    non_empty_idx = np.where(np.diff(targets.indptr) != 0)[0]
    num_empty = targets.shape[0] - len(non_empty_idx)
    if num_empty > 0 and verbose:
        print('\t\tWarning: \'targets\' has {} empty rows. Is that ok?'.format(num_empty))

    for u in non_empty_idx:

        # sort all query output values (argsort sorts low to high)
        scores_u_idx_sort = np.argsort(scores[u])[::-1]

        # find all the positions where a new relevant item is found
        targets_u_idx = targets[u].indices
        is_hit = np.in1d(scores_u_idx_sort, targets_u_idx, assume_unique=True)

        # precision at points where recall changes, i.e., when a hit is found
        precision = 0.
        for flag, idx in enumerate(np.where(is_hit)[0]):
            precision += is_hit[:(idx + 1)].sum() / (idx + 1)
            # idx + 1 so that the slice includes the position idx + 1, and to
            # divide by the number of elems in the slice (python is 0-based)

            if flag == 0:
                rr.append(precision)

        # average precisions at recall points
        avgp.append(precision / len(targets_u_idx))

    return rr, avgp


def compute_precision_recall(scores, targets, k_list, verbose=True):
    """
    Compute the precision and recall of each target continuation in `targets`
    given the playlist-song `scores`, with different list lengths.

    Parameters
    ----------
    scores: np array, shape (num_playlists, num_songs)
        Probability or score for each (playlist, song) pair.
    targets: sparse csr_matrix, shape (num_playlists, num_songs)
        Binary sparse array indicating target playlist continuations.
    k_list: list
        Each item is a list length.
    verbose: bool
        Print info if True.

    Returns
    -------
    precision: dict of lists, lists of length num_playlists
        Lists with the precision of each continuation, indexed by list length.
    recall: dict of lists, lists of length num_playlists
        Lists with the recall of each continuation, indexed by list length.
    """

    if verbose:
        print('\tComputing precision and recall...')

    precision = {K: [] for K in k_list}
    recall = {K: [] for K in k_list}

    # find rows where target is not empty (https://mike.place/2015/sparse/)
    non_empty_idx = np.where(np.diff(targets.indptr) != 0)[0]
    num_empty = targets.shape[0] - len(non_empty_idx)
    if num_empty > 0 and verbose:
        print('\t\tWarning: target has {} empty rows. Is that ok?'.format(num_empty))

    for u in non_empty_idx:

        # sort all query output values (argsort sorts low to high)
        scores_u_idx_sort = np.argsort(scores[u])[::-1]

        # find all the positions where a new relevant item is found
        targets_u_idx = targets[u].indices
        is_hit = np.in1d(scores_u_idx_sort, targets_u_idx, assume_unique=True)

        for K in k_list:
            num_hits = is_hit[:K].sum()
            precision[K].append(num_hits / K)
            recall[K].append(num_hits / len(targets_u_idx))

    return precision, recall


def compute_metrics(scores, targets, k_list, verbose=True):
    """
    Compute ranking-based metrics: rank, reciprocal rank, average precision and
    recall@K, for K in k_list.
    Note that the arrays are playlists x songs and not vice-versa.

    Parameters
    ----------
    scores: np array, shape (num_playlists, num_songs)
        Probability or score for each (playlist, song) pair.
    targets: sparse csr_matrix, shape (num_playlists, num_songs)
        Binary sparse array indicating target playlist continuations.
    k_list: list
        Each item is a list length.
    verbose: bool
        Print info if True.

    Returns
    -------
    song_metrics: tuple containing metrics
        rank: list of length targets.nnz
        rr: list of length num_playlists
        avgp: list of length num_playlists
        rec: dict with lists of length num_playlists for rec@{K for in k_list}
    """

    # provide data information
    playlists = np.sum(targets.sum(axis=1) != 0)
    songs = np.sum(targets.sum(axis=0) != 0)
    interactions = np.sum(targets)

    if verbose:
        print('\nComputing metrics ({} playlists, {} songs, {} interactions)'
              '...'.format(playlists, songs, interactions))

    # compute metrics
    rank = find_rank(scores, targets, verbose)
    rr, avgp = compute_average_precision(scores, targets, verbose)
    _, rec = compute_precision_recall(scores, targets, k_list, verbose)

    return rank, rr, avgp, rec


def summarize_metrics(rank, rr, avgp, rec, k_list, ci=False, pivotal=True,
                      verbose=True):
    """
    Summarize ranking-based metrics: median rank, mean reciprocal rank, mean
    average precision and mean recall@K, for K in k_list.

    Parameters
    ----------
    rank: list of length targets.nnz
        Rank achieved by every withheld song.
    rr: list of length num_playlists
        Reciprocal rank for every playlist extended.
    avgp: list of length num_playlists
        Average precision for every playlist extended.
    rec: dict with lists of length num_playlists for rec@K (K in k_list)
        Recall for every playlist extended.
    k_list: list
        Each item is a list length.
    ci: bool
        Compute basic Bootstrap confidence intervals if True.
    pivotal: bool
        Compute "pivotal" intervals if True, else "percentile" intervals.
    verbose: bool
        Print info if True.

    Returns
    -------
    metrics: tuple of floats
        Summarized metrics, possibly with lower and upper confidence values.
    """

    assert len(rr) == len(avgp) == len(rec[k_list[0]])

    if verbose:
        print('\nSummarizing metrics ({} playlists, {} interactions)'
              '...'.format(len(rr), len(rank)))

    if not ci:
        # report metrics without confidence intervals
        metrics = [np.median(rank), np.mean(rr), np.mean(avgp)]
        for K in k_list:
            metrics += [np.mean(rec[K])]

        if verbose:
            names = ['med_rank', 'mrr', 'map'] + ['mean_rec{}'.format(K) for K in k_list]
            print(('\n\t\t' + '{:<15}' * (3 + len(k_list))).format(*names))
            print(('\t\t' + '{:<15.1f}' * 1 + '{:<15.2%}' * (2 + len(k_list))).format(*metrics))
            print('')

    else:
        # report metrics with basic Bootstrap confidence intervals
        # bootstrapped defaults alpha=0.05, num_iterations=10k

        if not rank:
            metrics = [np.nan] * 18

        else:
            # median rank
            metric_ci = bs.bootstrap(values=np.array(rank), stat_func=bs_stats.median, is_pivotal=pivotal)
            metrics = [metric_ci.value, metric_ci.lower_bound, metric_ci.upper_bound]

            # mean reciprocal rank, average precision and recall at K
            for metric in [rr, avgp] + [rec[K] for K in k_list]:
                metric_ci = bs.bootstrap(values=np.array(metric), stat_func=bs_stats.mean, is_pivotal=pivotal)
                metrics += [metric_ci.value, metric_ci.lower_bound, metric_ci.upper_bound]

        if verbose:
            temp = []
            for K in k_list:
                temp += ['mean_rec{}'.format(K), 'mean_rec{}_l'.format(K), 'mean_rec{}_u'.format(K)]
            names = ['med_rank', 'med_rank_l', 'med_rank_u',
                     'mrr', 'mrr_l', 'mrr_u',
                     'map', 'map_l', 'map_u'] + temp

            print(('\n\t\t' + '{:<15}' * (9 + 3 * len(k_list))).format(*names))
            print(('\t\t' + '{:<15.1f}' * 3 + '{:<15.2%}' * (6 + 3 * len(k_list))).format(*metrics))
            print('')

    return metrics


def evaluate(scores, targets, queries, train_occ, k_list, ci=False,
             pivotal=True, song_occ=None, metrics_file=None):
    """
    Evaluate continuations induced by `scores` given target `targets`.
    The arguments are lists, and each item in a list corresponds to one run
    of the playlist continuation model, typically the run of one fold.
    Note that the arrays are playlists x songs and not vice-versa.

    Parameters
    ----------
    scores: list of numpy arrays of shape (num_playlists, num_songs)
        Probability or score for each (playlist, song) pair.
    targets: list of sparse csr_matrix's of shape (num_playlists, num_songs)
        Binary sparse array indicating target playlist continuations.
    queries: list of sparse csr_matrix's of shape (num playlists, num_songs)
        Binary sparse array indicating playlist queries.
    train_occ: list of numpy arrays of shape (num_songs, )
        Song occurrences when the model used to predict `scores` was trained.
    k_list: list
        Each item is a list length.
    ci: bool
        Compute basic Bootstrap confidence intervals if True.
    pivotal: bool
        Compute "pivotal" intervals if True, else "percentile" intervals.
    song_occ: list
        Test on songs observed `song_occ` times during model training.
    metrics_file: str
        File path to save ranks and summarized metrics.
    """

    print('\nEvaluating playlist continuations...')

    # mask `scores` corresponding to playlist queries
    for i in range(len(scores)):
        mask_array_items(scores[i], queries[i])

    # evaluate predictions given target continuations
    rank, rr, avgp, rec = [], [], [], {K: [] for K in k_list}
    for i in range(len(scores)):
        rank_i, rr_i, avgp_i, rec_i = compute_metrics(scores[i], targets[i], k_list)
        rank += rank_i
        rr += rr_i
        avgp += avgp_i
        for K in k_list:
            rec[K] += rec_i[K]
    metrics = summarize_metrics(rank, rr, avgp, rec, k_list, ci=ci, pivotal=pivotal)
    if metrics_file is not None:
        np.savetxt(metrics_file + '_all_songs.rank', rank)
        np.savetxt(metrics_file + '_all_songs.rr', rr)
        np.savetxt(metrics_file + '_all_songs.rec', metrics[3:])

    # conduct cold-start analysis
    if song_occ is not None:

        # for all but last, keep songs observed exactly song_obs times
        for occ in song_occ[:-1]:
            rank, rr, avgp, rec = [], [], [], {K: [] for K in k_list}
            for i in range(len(scores)):
                print('\nKeep songs observed {} at training...'.format(occ))
                target_i = copy.deepcopy(targets[i])
                mask_array_cols(target_i, np.where(train_occ[i] != occ)[0])
                rank_i, rr_i, avgp_i, rec_i = compute_metrics(scores[i], target_i, k_list)
                rank += rank_i
                rr += rr_i
                avgp += avgp_i
                for K in k_list:
                    rec[K] += rec_i[K]
            summarize_metrics(rank, rr, avgp, rec, k_list, ci=ci, pivotal=pivotal)

        # for the last, keep songs observed song_occ+ times, included
        occ = song_occ[-1]
        rank, rr, avgp, rec = [], [], [], {K: [] for K in k_list}
        for i in range(len(scores)):
            print('\nKeep songs observed {}+ (incl.) times at training...'.format(occ))
            target_i = copy.deepcopy(targets[i])
            mask_array_cols(target_i, np.where(train_occ[i] < occ)[0])
            rank_i, rr_i, avgp_i, rec_i = compute_metrics(scores[i], target_i, k_list)
            rank += rank_i
            rr += rr_i
            avgp += avgp_i
            for K in k_list:
                rec[K] += rec_i[K]
        summarize_metrics(rank, rr, avgp, rec, k_list, ci=ci, pivotal=pivotal)

        # for the last, keep songs observed song_occ- times, not included
        occ = song_occ[-1]
        rank, rr, avgp, rec = [], [], [], {K: [] for K in k_list}
        for i in range(len(scores)):
            print('\nKeep songs observed {}- (not incl.) times at training...'.format(occ))
            target_i = copy.deepcopy(targets[i])
            mask_array_cols(target_i, np.where(train_occ[i] >= occ)[0])
            rank_i, rr_i, avgp_i, rec_i = compute_metrics(scores[i], target_i, k_list)
            rank += rank_i
            rr += rr_i
            avgp += avgp_i
            for K in k_list:
                rec[K] += rec_i[K]
        summarize_metrics(rank, rr, avgp, rec, k_list, ci=ci, pivotal=pivotal)

        # for comparability between hybrid and pure collaborative systems,
        # compare results for songs with 1+ occurrences
        rank, rr, avgp, rec = [], [], [], {K: [] for K in k_list}
        for i in range(len(scores)):
            print('\nKeep songs observed 1+ (incl.) times at training...')
            target_i = copy.deepcopy(targets[i])
            mask_array_cols(target_i, np.where(train_occ[i] == 0)[0])
            rank_i, rr_i, avgp_i, rec_i = compute_metrics(scores[i], target_i, k_list)
            rank += rank_i
            rr += rr_i
            avgp += avgp_i
            for K in k_list:
                rec[K] += rec_i[K]
        metrics = summarize_metrics(rank, rr, avgp, rec, k_list, ci=ci, pivotal=pivotal)

        if metrics_file is not None:
            np.savetxt(metrics_file + '_inset_songs.rank', rank)
            np.savetxt(metrics_file + '_inset_songs.rr', rr)
            np.savetxt(metrics_file + '_inset_songs.rec', metrics[3:])
