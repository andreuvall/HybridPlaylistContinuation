from __future__ import print_function
from __future__ import division

from utils.data import load_data, compute_playlists_coo_stats, shape_data
from utils.mf import select_model, show_design, fit, test

import numpy as np

import argparse
import os

'''
Collaborative Filtering baseline for music playlist continuation based on the
factorization of the playlist-song matrix of co-occurrences. We use the
Weighted Matrix Factorization algorithm. Once the factorization is learned, it
can be used to populate a matrix of playlist-song scores describing how well a
playlist and a song fit together. Thus, a playlist can be extended by
selecting the songs with highest score. This approach is purely collaborative
in the usual sense in the recommender systems literature, i.e., it only relies
on the collaborative information given by the actual playlists examples.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Collaborative music playlist continuation based on matrix factorization.')
    parser.add_argument('--config', type=str, help='model configuration file', default='config/cf/wmf.py', metavar='')
    parser.add_argument('--fit', action='store_true', help='factorize the training matrix of \"query\" playlists vs songs')
    parser.add_argument('--test', action='store_true', help='evaluate the playlist extensions yielded by the factorization')
    parser.add_argument('--song_obs', type=int, help='test on songs observed song_obs+ times during training', default=None, metavar='')
    args = parser.parse_args()

    # set model configuration
    model = select_model(args.config)

    # load data: playlists, splits and artist info
    data_dir = 'data/AotM-2011'
    playlists_coo, split_idx, _, song2artist = load_data(data_dir, model)

    # playlists_coo are the playlists stored in coordinate format
    playlists_idx, songs_idx, idx2song = playlists_coo

    # each playlist is split into a "query" of ~80% of the songs (train_idx +
    # valid_idx) and a "continuation" of ~20% of the songs (test_idx)
    train_idx, valid_idx, test_idx = split_idx

    # prepare output directory
    out_dir = os.path.join('results', model.mode, model.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # provide model information
    print('\nMatrix Factorization:')
    show_design(model)

    # provide data information
    print('\nTraining split:')
    full_train_idx = np.hstack((train_idx, valid_idx))
    compute_playlists_coo_stats(
        playlists_idx[full_train_idx], songs_idx[full_train_idx],
        idx2song, song2artist
    )
    print('\nTest split:')
    compute_playlists_coo_stats(
        playlists_idx[test_idx], songs_idx[test_idx], idx2song, song2artist
    )

    if args.fit:
        #
        # use the "query" playlists in the training split as learning examples
        #

        # prepare playlist-song matrix at training

        # note that shape_data yields the song-playlist matrix (transposed)
        # also we use mode='test' because MF needs the full matrix shape
        _, Y = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            mode='test', subset=full_train_idx, verbose=True
        )

        # transpose to obtain the playlist-song matrix
        Y = Y.T.tocsr()

        # factorize the playlist-song matrix
        fit(model, Y, out_dir)

    if args.test:
        #
        # extend the "query" playlists from the training split using the scores
        # given by the matrix factorization and evaluate these continuations by
        # comparing them to the actual with-held continuations
        #

        # prepare playlist-song matrix at test

        # note that shape_data yields the song-playlist matrix (transposed)
        # also we use mode='test' because MF needs the full matrix shape
        _, Y = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            mode='test', subset=test_idx, verbose=True
        )

        # transpose to obtain the playlist-song matrix
        Y = Y.T.tocsr()

        # songs in the "query" playlists need to be masked to make sure that
        # they are not recommended as continuations
        _, Y_train = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            mode='test', subset=full_train_idx
        )

        # transpose to obtain the playlist-song matrix
        Y_train = Y_train.T.tocsr()

        # evaluate the continuations given by the matrix factorization
        test(model=model, test_target=Y, train_target=Y_train, out_dir=out_dir,
             song_obs=args.song_obs)
