from __future__ import print_function
from __future__ import division

from sklearn import preprocessing as prep

from utils.data import load_data, compute_playlists_coo_stats, shape_data
from utils.song2play import select_model, show_design, fit, test

import theano
import numpy as np

import argparse
import os

'''
Hybrid music playlist continuation based on a song-to-playlist classifier.
We learn a classifier that takes song features as inputs and predicts the
playlists songs belong to. Once it is learned, such classifier can be
used to populate a matrix of song-playlist scores describing how well a song
and a playlist fit together. Thus, a playlist can be extended by selecting
the songs with highest score. This approach is "hybrid" in the usual sense in
the recommender systems literature, i.e., it combines content (given by the
song features) and cf information (given by playlists examples).
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hybrid music playlist continuation based on a song-to-playlist classifier.')
    parser.add_argument('--config', type=str, help='model specification file', default='config/hybrid/standalone/logs.py', metavar='')
    parser.add_argument('--fit', action='store_true', help='fit the classifier on training \"query\" playlists')
    parser.add_argument('--test', action='store_true', help='evaluate the playlist extensions yielded by the classifier')
    parser.add_argument('--song_obs', type=int, help='test on songs observed song_obs+ times during training', default=None, metavar='')
    args = parser.parse_args()

    # set model configuration
    model = select_model(args.config)

    # load data: playlists, splits, features and artist info
    data_dir = 'data/AotM-2011'
    playlists_coo, split_idx, features, song2artist = load_data(data_dir, model)

    # playlists_coo are the playlists stored in coordinate format
    playlists_idx,  songs_idx, idx2song = playlists_coo

    # each playlist is split into a "query" of ~80% of the songs (train_idx +
    # valid_idx) and a "continuation" of ~20% of the songs (test_idx)
    train_idx, valid_idx, test_idx = split_idx

    # prepare output directory
    out_dir = os.path.join('results', model.mode, model.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # provide model information
    print('\nNetwork:')
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

        # prepare input song features and playlist targets at training
        X, Y = shape_data(playlists_idx, songs_idx, idx2song, features,
                          mode='fit', subset=full_train_idx, verbose=True)

        # preprocess input features if required
        if model.standardize:
            X = prep.robust_scale(X)

        if model.normalize:
            X = prep.normalize(X, norm=model.normalize)

        # fit the classifier
        fit(
            model=model,
            train_input=X.astype(theano.config.floatX),
            train_target=Y.astype(np.int8),
            out_dir=out_dir
        )

    if args.test:
        #
        # extend the "query" playlists from the training split using the scores
        # given by the hybrid classifier and evaluate these continuations
        # by comparing them to the actual with-held continuations
        #

        # prepare input song features and playlist targets at test
        X, Y = shape_data(playlists_idx, songs_idx, idx2song, features,
                          mode='test', subset=test_idx, verbose=True)

        # preprocess input features if required
        if model.standardize:
            # use the training song features to standardize the test data
            X_prep, _ = shape_data(playlists_idx, songs_idx, idx2song, features,
                                   mode='fit', subset=full_train_idx)
            scaler = prep.RobustScaler()
            scaler.fit(X_prep)
            X = scaler.transform(X)

        if model.normalize:
            X = prep.normalize(X, norm=model.normalize)

        # songs in the "query" playlists need to be masked to make sure that
        # they are not recommended as continuations
        _, Y_train = shape_data(playlists_idx, songs_idx, idx2song, features,
                                mode='test', subset=full_train_idx)

        # evaluate the continuations given by the classifier
        test(
            model=model,
            test_input=X.astype(theano.config.floatX),
            test_target=Y.astype(np.int8),
            train_target=Y_train.astype(np.int8),
            out_dir=out_dir,
            song_obs=args.song_obs
        )
