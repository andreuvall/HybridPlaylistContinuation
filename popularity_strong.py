from __future__ import print_function
from __future__ import division

from utils.data import load_data, show_data_splits, shape_data
from utils.evaluation import evaluate

import argparse
import os

import numpy as np


'''
Popularity baseline for music playlist continuation. (It is also possible to
simply compute a random baseline giving the --random option.)

In this program we explore the so-called strong generalization setting. That is,
the song popularity is computed on playlists different than those extended.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Popularity for music playlist continuation.')
    parser.add_argument('--dataset', type=str, help='path to the playlists dataset directory', metavar='')
    parser.add_argument('--msd', type=str, help='path to the MSD directory', metavar='')
    parser.add_argument('--ci', action='store_true', help='compute confidence intervals if True')
    parser.add_argument('--song_occ', type=int, help='test on songs observed song_occ times during training', nargs='+', metavar='')
    parser.add_argument('--metrics_file', type=str, help='file name to save metrics', metavar='')
    parser.add_argument('--random', action='store_true', help='forget about pop, evaluate a random baseline')
    args = parser.parse_args()

    if args.random:
        # set randomness if computing random baseline
        print('This is not pop but a random baseline... you sure?')
        rng = np.random.RandomState(1)

    # load data: playlists, splits and artist info
    data_name = os.path.basename(os.path.normpath(args.dataset))
    data = load_data(args.dataset, args.msd, None)
    playlists_coo, split_weak, split_strong, features, song2artist = data

    # playlists_coo are the playlists stored in coordinate format
    playlists_idx, songs_idx, _, idx2song = playlists_coo

    # split_weak provides a query/continuation split
    train_idx_cnt, test_idx_cnt = np.hstack(split_weak[:2]), split_weak[2]

    cont_output_l, Y_cont_l, Y_query_l, train_occ_l = [], [], [], []
    for fold in range(5):

        print('\nRunning fold {}...'.format(fold))

        # split_strong defines a playlist-disjoint split
        # chose one of the folds
        fold_strong = split_strong[fold]
        train_idx_dsj, test_idx_dsj = np.hstack(fold_strong[:2]), fold_strong[2]

        # define splits for this experiment
        # train model on the intersection of disjoint training and queries
        # validate model on the inters. of disjoint training and continuations
        # fit the model on the disjoint training playlists
        # extend only the playlist-disjoint test split
        train_idx = np.intersect1d(train_idx_dsj, train_idx_cnt)
        valid_idx = np.intersect1d(train_idx_dsj, test_idx_cnt)
        fit_idx = train_idx_dsj
        query_idx = np.intersect1d(test_idx_dsj, train_idx_cnt)
        cont_idx = np.intersect1d(test_idx_dsj, test_idx_cnt)

        # provide data information
        show_data_splits(playlists_idx, songs_idx, idx2song, song2artist,
                         train_idx, valid_idx, fit_idx, query_idx, cont_idx)

        #
        # extend the playlists in the query split and evaluate the
        # continuations by comparing them to actual withheld continuations
        #

        # prepare song-playlist matrix in test continuations
        _, Y_cont = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=cont_idx
        )

        # prepare song-playlist matrix in test queries
        # used to mask songs from queries and to calculate playlist factors
        _, Y_query = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=query_idx
        )

        # calculate number of song occurrences in "fit" playlists
        # used for cold-start analysis
        _, Y_fit = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=fit_idx
        )
        train_occ = np.asarray(Y_fit.sum(axis=1)).flatten()

        # predict song-playlist scores
        print('\nPredicting song-playlist scores...')
        cont_output = np.repeat(train_occ.reshape(-1, 1), Y_cont.shape[1], axis=1)

        if args.random:
            # overwrite cont_output if we need a random baseline
            cont_output = rng.rand(*Y_cont.shape)

        # append arrays re-shaping for evaluation
        cont_output_l.append(cont_output.T)
        Y_cont_l.append(Y_cont.T.tocsr())
        Y_query_l.append(Y_query.T.tocsr())
        train_occ_l.append(train_occ)

    # evaluate the continuations
    evaluate(
        scores=cont_output_l,
        targets=Y_cont_l,
        queries=Y_query_l,
        train_occ=train_occ_l,
        k_list=[10, 30, 100],
        ci=args.ci,
        song_occ=args.song_occ,
        metrics_file=args.metrics_file
    )
