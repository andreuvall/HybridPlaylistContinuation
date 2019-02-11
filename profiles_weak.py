from __future__ import print_function
from __future__ import division

from sklearn.utils import check_random_state
from sklearn import preprocessing as prep

from utils.data import load_data, show_data_splits, shape_data
from utils.evaluation import evaluate
from utils.profiles import select_model, show_design, train, fit, compute_scores

import theano
import lasagne as lg
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

As it is, this approach only works on the so-called weak generalization setting. 
That is, the model is trained on the same playlists that will be extended.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hybrid music playlist continuation based on a song-to-playlist classifier.')
    parser.add_argument('--model', type=str, help='path to the model specification file', metavar='')
    parser.add_argument('--dataset', type=str, help='path to the playlists dataset directory', metavar='')
    parser.add_argument('--msd', type=str, help='path to the MSD directory', metavar='')
    parser.add_argument('--train', action='store_true', help='train the song-to-playist classifier with monitoring')
    parser.add_argument('--fit', action='store_true', help='fit the song-to-playlist classifier')
    parser.add_argument('--test', action='store_true', help='evaluate the playlist continuations')
    parser.add_argument('--ci', action='store_true', help='compute confidence intervals if True')
    parser.add_argument('--song_occ', type=int, help='test on songs observed song_occ times during training', nargs='+', metavar='')
    parser.add_argument('--metrics_file', type=str, help='file name to save metrics', metavar='')
    parser.add_argument('--seed', type=int, help='set random behavior', metavar='')
    args = parser.parse_args()

    # set random behavior
    rng = check_random_state(args.seed)
    lg.random.set_rng(rng)

    # set model configuration
    model = select_model(args.model)

    # prepare output directory
    data_name = os.path.basename(os.path.normpath(args.dataset))
    out_dir = os.path.join('params', 'profiles', model.name + '_' + data_name + '_weak')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data: playlists, splits, features and artist info
    data = load_data(args.dataset, args.msd, model)
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

    # provide model information
    print('\nNetwork:')
    show_design(model)

    if args.train:
        #
        # train the hybrid model while validating on withheld playlists
        #

        # prepare input song features and playlist targets at training
        X_train, Y_train = shape_data(
            playlists_idx, songs_idx, idx2song, features,
            mode='train', subset=train_idx
        )

        # prepare input song features and playlist targets at validation
        X_valid, Y_valid = shape_data(
            playlists_idx, songs_idx, idx2song, features,
            mode='test', subset=valid_idx
        )

        # preprocess input features if required
        # use the training song features to standardize the validation data
        if model.standardize:
            scaler = prep.RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

        if model.normalize:
            X_train = prep.normalize(X_train, norm=model.normalize)
            X_valid = prep.normalize(X_valid, norm=model.normalize)

        # train the classifier
        train(
            model=model,
            train_input=X_train.astype(theano.config.floatX),
            train_target=Y_train.astype(np.int8),
            valid_input=X_valid.astype(theano.config.floatX),
            valid_target=Y_valid.astype(np.int8),
            out_dir=out_dir,
            random_state=rng
        )

    if args.fit:
        #
        # fit the hybrid model
        #

        # prepare input song features and playlist targets at training
        X_fit, Y_fit = shape_data(
            playlists_idx, songs_idx, idx2song, features,
            mode='train', subset=fit_idx
        )

        # preprocess input features if required
        if model.standardize:
            X_fit = prep.robust_scale(X_fit)

        if model.normalize:
            X_fit = prep.normalize(X_fit, norm=model.normalize)

        # fit the classifier
        fit(
            model=model,
            fit_input=X_fit.astype(theano.config.floatX),
            fit_target=Y_fit.astype(np.int8),
            out_dir=out_dir,
            random_state=rng
        )

    if args.test:
        #
        # extend the playlists in the query split and evaluate the
        # continuations by comparing them to actual withheld continuations
        #

        # prepare input song features and playlist targets at test
        X_cont, Y_cont = shape_data(
            playlists_idx, songs_idx, idx2song, features,
            mode='test', subset=cont_idx
        )

        # preprocess input features if required
        # use the training song features to standardize the test data
        if model.standardize:
            X_fit, _ = shape_data(
                playlists_idx, songs_idx, idx2song, features,
                mode='train', subset=fit_idx
            )
            scaler = prep.RobustScaler()
            scaler.fit(X_fit)
            X_cont = scaler.transform(X_cont)

        if model.normalize:
            X_cont = prep.normalize(X_cont, norm=model.normalize)

        # songs in the "query" playlists need to be masked to make sure that
        # they are not recommended as continuations
        _, Y_query = shape_data(
            playlists_idx, songs_idx, idx2song, features,
            mode='test', subset=query_idx
        )

        # get number of song occurrences when fitting for cold-start analysis
        # Y_fit = Y_query
        train_occ = np.asarray(Y_query.sum(axis=1)).flatten()

        # compute the song-playlist scores
        cont_output = compute_scores(
            model=model,
            params_dir=out_dir,
            cont_input=X_cont.astype(theano.config.floatX),
            cont_target=Y_cont.astype(np.int8)
        )

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
