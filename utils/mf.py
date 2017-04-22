# Collaborative filtering baseline utils.

from __future__ import print_function
from __future__ import division

from evaluation import *

import implicit
import numpy as np

import cPickle
import time
import os
import sys


def select_model(model_path):
    """ Select model and related functions. """

    dir_str = os.path.dirname(model_path)
    if dir_str.split('/')[1] != 'cf':
        sys.exit('\nThe configuration files passed to playlist_cf.py must be '
                 'located in the directory config/cf. The provided file resides '
                 'in {}.'.format(dir_str))

    model = False
    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]
    exec ('from config.cf import {} as model'.format(model_str))

    model.mode = 'cf'
    model.feat_mode = None
    model.name = model_str

    return model


def show_design(model):
    """ Print details contained in a specification file. """

    print(
        '\tTraining options\n'
        '\tnum_factors = {}\n'
        '\tmax_epochs = {}\n'
        '\tpositive_weight = {}\n'
        '\tL2_weight = {}'.format(model.num_factors, model.max_epochs,
                                  model.positive_weight, model.L2_weight)
    )


def fit(model, train_target, out_dir):
    """
    Factorize the training matrix of playlist-song pairs. Return nothing.

    Parameters
    ----------
    model: model file
        Model specification.
    train_target: numpy array, shape (num playlists, num songs)
        Matrix of playlist-song co-occurrences at the train split.
    out_dir: string
        Path to the results directory
    """

    print('\nSetting up fit...')

    # identify dimensions
    num_playlists, num_songs = train_target.shape

    # initialize weights
    playlists = np.random.rand(num_playlists, model.num_factors) * 0.01
    songs = np.random.rand(num_songs, model.num_factors) * 0.01

    print('\nFitting...')

    for epoch in xrange(1, model.max_epochs + 1):

        # keep track of time
        start_time = time.time()

        playlists, songs = implicit.alternating_least_squares(
            Cui=model.positive_weight * train_target,
            factors=model.num_factors,
            X=playlists,
            Y=songs,
            iterations=1,
            regularization=model.L2_weight,
            use_cg=False
        )

        print('\tEpoch {} of {} took {:.3f}s'.format(
            epoch, model.max_epochs, time.time() - start_time)
        )

    # save the fit model
    print('\nSaving model weights...')

    params = (playlists, songs)
    params_file = '{}_params.pkl'.format(model.name)
    with open(os.path.join(out_dir, params_file), 'w') as f:
        cPickle.dump(params, f)


def test(model, test_target, train_target, out_dir, song_obs=None):
    """
    Evaluate the playlist continuations given by the latent factors
    obtained from factorizing the playlist-song matrix.

    Parameters
    ----------
    model: model file
        Model specification.
    test_target: numpy array, shape (num playlists, num songs)
        Matrix of playlist-song co-occurrences at the test split.
    train_target: numpy array, shape (num playlists, num_songs)
        Matrix of playlist-song co-occurrences at the train split.
    out_dir: string
        Path to the results directory
    song_obs: int
        Test on songs observed song_obs times during training.
    """

    print('\nSetting up test...')

    # load previously learned latent factors
    print('\nLoading fit weights to the model...')

    params_file = '{}_params.pkl'.format(model.name)
    if os.path.isfile(os.path.join(out_dir, params_file)):
        with open(os.path.join(out_dir, params_file), 'rb') as f:
            playlists, songs = cPickle.load(f)
    else:
        sys.exit('\tThe file {} does not exist yet. You need to fit the model '
                 'first.'.format(os.path.join(out_dir, params_file)))

    # use the latent factors to populate a matrix of playlist-song scores
    print('\nPredicting playlist-song scores...')
    test_output = playlists.dot(songs.T)

    # use the scores to extend query playlists
    print('\nEvaluating playlist continuations...')

    # mask known good continuations from training
    mask_training_targets(test_output, train_target, verbose=True)

    # keep only songs with song_obs observations at training time
    if song_obs is not None:
        occ = np.array(train_target.sum(axis=0)).flatten()
        test_target[:, np.where(occ != song_obs)[0]] = 0
        test_target.eliminate_zeros()

    # find rank of actual continuations
    song_rank = find_rank(test_output, test_target, verbose=False)

    # compute mean average precision
    song_avgp = compute_map(test_output, test_target, verbose=False)

    # compute precision recall
    song_prc, song_rec = compute_precision_recall(
        test_output, test_target, [10, 30, 100], verbose=False)
    song_metrics = [np.median(song_rank), np.mean(song_avgp)]
    for K in [10, 30, 100]:
        song_metrics += [np.mean(song_rec[K])]

    metrics = ['med_rank', 'map', 'mean_rec10', 'mean_rec30', 'mean_rec100']
    print(('\n\t' + '{:<13}' * 5).format(*metrics))
    print(('\t' + '{:<13.1f}' * 1 + '{:<13.2%}' * 4).format(*song_metrics))
