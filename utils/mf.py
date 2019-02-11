# Collaborative filtering utils.

from __future__ import print_function
from __future__ import division

from tqdm import tqdm

from utils.evaluation import mask_array_items, mask_array_rows, compute_metrics, summarize_metrics
from implicit import _als

import implicit
import numpy as np

import cPickle
import time
import copy
import os
import sys

EVERY = 10


def select_model(model_path):
    """ Select model and related functions. """

    cfg_dir, model_dir = os.path.dirname(model_path).split('/')
    model_file = os.path.basename(model_path).split('.py')[0]

    model = False
    exec ('from {}.{} import {} as model'.format(cfg_dir, model_dir, model_file))
    model.name = model_file

    return model


def show_design(model):
    """ Print details contained in a specification file. """

    print(
        '\tTraining options\n'
        '\tnum_factors = {}\n'
        '\tmax_epochs = {}\n'
        '\tpositive_weight = {}\n'
        '\tl2_weight = {}'.format(model.num_factors, model.max_epochs,
                                  model.positive_weight, model.l2_weight)
    )


def load_mf(params_path, verbose=True):
    """ Load factorization-based song and playlist factors. """

    if verbose:
        print('\nLoading factorization-based song and playlist factors...')

    if not os.path.isfile(params_path):
        sys.exit('\tThe file {} does not exist yet. You need to fit the model '
                 'first.'.format(params_path))
    else:
        with open(params_path, 'rb') as f:
            songs, playlists = cPickle.load(f)

    return songs, playlists


def load_audio2cf(audio2cf_path, idx2song, verbose=True):
    """ Load audio2cf-based song factors and shape them into an array. """

    if verbose:
        print('\nLoading audio2cf-based song factors...')

    if not os.path.isfile(audio2cf_path):
        sys.exit('\tThe file {} does not exist yet. You need to fit the model '
                 'first.'.format(audio2cf_path))

    else:
        with open(audio2cf_path, 'rb') as f:
            songs_d = cPickle.load(f)

        # build song2idx from idx2song
        song2idx = {v: k for k, v in idx2song.iteritems()}
        X = np.vstack([songs_d[idx2song[idx]] for idx in unique_songs])
        songs = None

    return songs


def train(model, train_target, valid_target, out_dir, use_gpu=False):
    """
    Factorize the training matrix of song-playlist pairs. Monitor on a
    validation dataset. Return nothing.

    Parameters
    ----------
    model: model file
        Model specification.
    train_target: numpy array, shape (num songs, num playlists)
        Target array of playlists the songs belong to for training.
    valid_target: numpy array, shape (num songs, num playlists)
        Target array of playlists the songs belong to for validation.
    out_dir: string
        Path to the params and logging directory
    use_gpu: bool
        Use GPU if True.
    """

    print('\nSetting up training...')

    # initialize model
    mf = implicit.als.AlternatingLeastSquares(
        factors=model.num_factors,
        regularization=model.l2_weight,
        use_cg=False,
        use_gpu=use_gpu,
        iterations=1
    )

    # set up metrics monitoring
    metrics = ['cost', 'med_rank', 'mrr', 'map', 'mean_rec10', 'mean_rec30', 'mean_rec100']
    train_log = {metric: [] for metric in metrics}
    valid_log = {metric: [] for metric in metrics}
    file_log = '{}_log_train.pkl'.format(model.name)

    # mask valid targets corresponding to unknown songs when MF is trained
    train_occ = np.asarray(train_target.sum(axis=1)).flatten()
    mask_array_rows(valid_target, np.where(train_occ == 0)[0])

    # train the classifier
    print('\nTraining...')

    for epoch in tqdm(xrange(1, model.max_epochs + 1)):

        mf.fit(item_users=model.positive_weight * train_target, show_progress=False)

        # keep track of the training cost (0 is for regularization=0)
        train_cost = _als.calculate_loss(
            train_target.astype(np.float), mf.item_factors, mf.user_factors, 0
        )
        train_log['cost'].append(train_cost)

        # keep track of the validation cost (0 is for regularization=0)
        valid_cost = _als.calculate_loss(
            valid_target.astype(np.float), mf.user_factors, mf.item_factors, 0
        )
        valid_log['cost'].append(valid_cost)

        if epoch % EVERY == 0:

            # compute training ranking metrics
            output = mf.item_factors.dot(mf.user_factors.T)
            train_metrics = compute_metrics(output.T, train_target.T.tocsr(), k_list=[10, 30, 100], verbose=False)
            train_metrics = summarize_metrics(*train_metrics, k_list=[10, 30, 100], verbose=False)

            # compute validation ranking metrics
            # mask output values not to recommend songs from query playlists
            mask_array_items(a=output, mask=train_target, verbose=False)
            valid_metrics = compute_metrics(output.T, valid_target.T.tocsr(), k_list=[10, 30, 100], verbose=False)
            valid_metrics = summarize_metrics(*valid_metrics, k_list=[10, 30, 100], verbose=False)

            tqdm.write(('\n\t' + '{:<13}' + '{:<13}' * 6).format('split', *metrics[1:]))
            tqdm.write(('\t' + '{:<13}' + '{:<13.1f}' * 1 + '{:<13.2%}' * 5).format('train', *train_metrics))
            tqdm.write(('\t' + '{:<13}' + '{:<13.1f}' * 1 + '{:<13.2%}' * 5).format('valid', *valid_metrics))
            tqdm.write('')

            for m, tm, vm in zip(metrics[1:], train_metrics, valid_metrics):
                train_log[m].append(tm)
                valid_log[m].append(vm)

    # save metrics and best setting
    with open(os.path.join(out_dir, file_log), 'wb') as f:
        cPickle.dump((train_log, valid_log), f)


def fit(model, fit_target, out_dir, fold=None, use_gpu=False):
    """
    Factorize the training matrix of song-playlist pairs. Return nothing.

    Parameters
    ----------
    model: model file
        Model specification.
    fit_target: numpy array, shape (num songs, num playlists)
        Matrix of song-playlist co-occurrences at the train split.
    out_dir: string
        Path to the params and logging directory
    fold: int or None
        Which fold is being fit. None if there aren't folds (e.g., in weak).
    use_gpu: bool
        Use GPU if true.
    """

    print('\nSetting up fit...')

    # initialize model
    mf = implicit.als.AlternatingLeastSquares(
        factors=model.num_factors,
        regularization=model.l2_weight,
        use_cg=False,
        use_gpu=use_gpu,
        iterations=1
    )

    # set up metrics monitoring and params file
    metrics = ['cost', 'med_rank', 'mrr', 'map', 'mean_rec10', 'mean_rec30', 'mean_rec100']
    log = {metric: [] for metric in metrics}
    if fold is None:
        log_file = '{}_log_fit.pkl'.format(model.name)
        params_file = '{}_params.pkl'.format(model.name)
    else:
        log_file = '{}_log_fit{}.pkl'.format(model.name, fold)
        params_file = '{}_params{}.pkl'.format(model.name, fold)

    # fit the factorization
    print('\nFitting...')
    start = time.time()

    for epoch in tqdm(xrange(1, model.max_epochs + 1)):

        mf.fit(item_users=model.positive_weight * fit_target, show_progress=False)

        # compute cost (0 is for regularization=0)
        fit_cost = _als.calculate_loss(
            fit_target.astype(np.float), mf.item_factors, mf.user_factors, 0
        )
        log['cost'].append(fit_cost)

        if epoch % EVERY == 0:

            # compute ranking metrics
            output = mf.item_factors.dot(mf.user_factors.T)
            fit_metrics = compute_metrics(output.T, fit_target.T.tocsr(), k_list=[10, 30, 100], verbose=False)
            fit_metrics = summarize_metrics(*fit_metrics, k_list=[10, 30, 100], verbose=False)

            tqdm.write(('\n\t\t' + '{:<13}' + '{:<13}' * 6).format('split', *metrics[1:]))
            tqdm.write(('\t\t' + '{:<13}' + '{:<13.1f}' * 1 + '{:<13.2%}' * 5).format('train', *fit_metrics))
            tqdm.write('')

            for m, fm in zip(metrics[1:], fit_metrics):
                log[m].append(fm)

    print('\nTime fitting: {:.4f} sec.'.format(time.time() - start))

    # save metrics
    with open(os.path.join(out_dir, log_file), 'wb') as f:
        cPickle.dump(log, f)

    # save fit model
    print('\nSaving fit model weights...')
    params = (mf.item_factors, mf.user_factors)
    with open(os.path.join(out_dir, params_file), 'w') as f:
        cPickle.dump(params, f)
