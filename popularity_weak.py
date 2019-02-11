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

In this program we explore the so-called weak generalization setting. That is,
the song popularity is computed on the same playlists that will be extended.
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
    playlists_coo, split_weak, _, features, song2artist = data

    # playlists_coo are the playlists stored in coordinate format
    playlists_idx, songs_idx, _, idx2song = playlists_coo

    # each playlist is split into a "query" of ~80% of the songs (train_idx +
    # valid_idx) and a "continuation" of ~20% of the songs (test_idx)
    train_idx, valid_idx, test_idx = split_weak

    # define splits for this experiment
    # train model on the training queries
    # validate model on the validation queries
    # fit the model on the full queries
    # extend all the playlists, using all queries and continuations
    train_idx = train_idx
    valid_idx = valid_idx
    fit_idx = np.hstack((train_idx, valid_idx))
    query_idx = fit_idx
    cont_idx = test_idx

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

    # calculate number of song occurrences in "query" playlists
    # used for cold-start analysis
    # fit_idx = query_idx --> Y_fit = Y_query
    train_occ = np.asarray(Y_query.sum(axis=1)).flatten()

    # predict song-playlist scores
    print('\nPredicting song-playlist scores...')
    cont_output = np.repeat(train_occ.reshape(-1, 1), Y_cont.shape[1], axis=1)

    if args.random:
        # overwrite cont_output if we need a random baseline
        cont_output = rng.rand(*Y_cont.shape)

    # evaluate the continuations
    evaluate(
        scores=[cont_output.T],
        targets=[Y_cont.T.tocsr()],
        queries=[Y_query.T.tocsr()],
        train_occ=[train_occ],
        k_list=[10, 30, 100],
        ci=args.ci,
        song_occ=args.song_occ,
        metrics_file=args.metrics_file
    )
