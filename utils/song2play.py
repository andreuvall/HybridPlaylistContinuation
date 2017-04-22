# Song-to-playlist classifier utils.

from __future__ import print_function
from __future__ import division

import theano
import theano.tensor as T

from sklearn.utils import shuffle

from evaluation import *

import lasagne as lg
import numpy as np

import cPickle
import time
import os
import sys


def select_model(model_path):
    """ Select model specification. """

    dir_str = os.path.dirname(model_path)
    if dir_str.split('/')[1] != 'hybrid':
        sys.exit('\nThe configuration files passed to playlist_hybrid.py must '
                 'be located in the directory config/hybrid. The provided file '
                 'resides in {}.'.format(dir_str))

    model = False
    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]
    exec ('from config.hybrid import {} as model'.format(model_str))

    model.mode = 'hybrid'
    model.name = model_str

    return model


def show_design(model):
    """ Print details contained in a specification file. """

    print(
        '\tStructure\n'
        '\tn_layers = {}\n'
        '\tn_hidden = {}\n'
        '\thid_nl = {}\n'
        '\tout_nl = {}\n\n'
        '\tTraining options\n'
        '\tbatch_size = {}\n'
        '\tlearning_rate = {}\n'
        '\tmax_epochs = {}\n'
        '\tmomentum = {}\n\n'
        '\tRegularization\n'
        '\tinput_dropout = {}\n'
        '\thidden_dropout = {}\n\n'
        '\tFeatures\n'
        '\tfeature = {}\n'
        '\tstandardize = {}\n'
        '\tnormalize = {}'.format(
            model.n_layers, model.n_hidden, model.hid_nl, model.out_nl,
            model.batch_size, model.learning_rate, model.max_epochs,
            model.momentum, model.input_dropout, model.hidden_dropout,
            model.feature, model.standardize, model.normalize)
    )


def build_model(feature_size, n_classes, model, verbose=True):
    """
    Build a feed forward neural net.

    Parameters
    ----------
    feature_size: int
        Dimensionality of the input features.
    n_classes: int
        Number of classes we want to classify into.
    model: model specification file
        Contains the model config.
    verbose: bool
        Print info if True.

    Returns
    -------
    input_layer: Lasagne layer
        Input layer.
    output_layer: Lasagne layer
        Output layer.
    """

    if verbose:
        print('\tBuilding FFNN...', end='')

    # input layer
    input_layer = lg.layers.InputLayer(shape=(None, feature_size))

    # dropout input units (rescale by default)
    input_layer_drop = lg.layers.DropoutLayer(
        incoming=input_layer,
        p=model.input_dropout
    )

    # hidden layer
    hidden_layer = lg.layers.batch_norm(
        lg.layers.DenseLayer(
            incoming=input_layer_drop,
            num_units=model.n_hidden,
            nonlinearity=getattr(lg.nonlinearities, model.hid_nl)
        )
    )

    # dropout hidden units (rescale by default)
    hidden_layer = lg.layers.DropoutLayer(
        incoming=hidden_layer,
        p=model.hidden_dropout
    )

    # stack n_layers - 1 more hidden layers
    for l in range(model.n_layers - 1):

        hidden_layer = lg.layers.batch_norm(
            lg.layers.DenseLayer(
                incoming=hidden_layer,
                num_units=model.n_hidden,
                nonlinearity=getattr(lg.nonlinearities, model.hid_nl)
            )
        )

        # dropout hidden units (rescale by default)
        hidden_layer = lg.layers.DropoutLayer(
            incoming=hidden_layer,
            p=model.hidden_dropout
        )

    # output layer
    output_layer = lg.layers.batch_norm(
        lg.layers.DenseLayer(
            incoming=hidden_layer,
            num_units=n_classes,
            nonlinearity=getattr(lg.nonlinearities, model.out_nl)
        )
    )

    # inform about the network size
    num_params = lg.layers.count_params(output_layer)
    if verbose:
        print(' [{} parameters]'.format(num_params))

    return input_layer, output_layer


def declare_theano_variables(output_layer, model, verbose=True):
    """
    Define target, network output, and cost as a function of them.

    Parameters
    ----------
    output_layer: Lasagne layer
        Output layer.
    model: model specification file
        Contains the model config.
    verbose: bool
        Print info if True.

    Returns
    -------
    target: Theano tensor
        Prediction target.
    stochastic_out: tuple
        Theano tensors for stochastic output and cost.
    deterministic_out: tuple
        Theano tensors for deterministic output and cost.
    """

    if verbose:
        print('\tDeclaring FFNN Theano variables...')

    # define target placeholder for the cost functions
    target = T.bmatrix('target')

    def define_cost(output_layer, target, model, determ):
        """
        Define Theano tensor for the cost as a function of the network output.
        The network output is also returned for convenience.

        Parameters
        ----------
        output_layer: Lasagne layer
            Output layer.
        target: Theano tensor
            Prediction target.
        model: model specification file
            Contains the model config.
        determ: bool
            Deterministic pass if True, else enable dropout.

        Returns
        -------
        output: Theano tensor
            Network output.
        cost: Theano tensor
            Cost as a function of output and target.
        """

        # Get network output
        output = lg.layers.get_output(output_layer, deterministic=determ)

        if model.out_nl == 'sigmoid':
            # Weighted BCE lets us put different trust in positive vs negative
            # observations (similar to weightedMF). The following holds if we
            # code t=1 for positive and t=0 for negative/not-known examples:
            # llh(example) = - w+ * t * log(p) - w- * (1 - t) * log(1 - p)
            cost = -1. * T.mean(
                model.positive_weight * target * T.log(output) +
                model.nonpositive_weight * (1. - target) * T.log(1. - output)
            )

        else:
            # categorical cross-entropy
            cost = T.mean(T.nnet.categorical_crossentropy(output, target))

        # regularize
        if model.L1_weight > 0:
            L1_reg = lg.regularization.regularize_network_params(output_layer, lg.regularization.l1)
            cost += model.L1_weight * L1_reg

        if model.L2_weight > 0:
            L2_reg = lg.regularization.regularize_network_params(output_layer, lg.regularization.l2)
            cost += model.L2_weight * L2_reg

        return output, cost

    # stochastic cost expression
    stochastic_out = define_cost(output_layer, target, model, determ=False)

    # deterministic cost expression
    deterministic_out = define_cost(output_layer, target, model, determ=True)

    return target, stochastic_out, deterministic_out


def compile_theano_functions(input_layer, output_layer, target, stochastic_out,
                             deterministic_out, exp, verbose=True):
    """
    Compile Theano functions for training, test and prediction.

    Parameters
    ----------
    input_layer: Lasagne layer
        Input layer.
    output_layer: Lasagne layer
        Output layer.
    target: Theano tensor
        Prediction target.
    stochastic_out: tuple
        Theano tensors for stochastic output and cost.
    deterministic_out: tuple
        Theano tensors for deterministic output and cost.
    model: model specification file
        Contains the model config.
    verbose: bool
        Print info if True.

    Returns
    -------
    train_model: Theano function
        Stochastic cost and output (with updates).
    test_model: Theano function
        Deterministic cost and output (without updates).
    predict_model: Theano function
        Deterministic output (without updates).
    """

    if verbose:
        print('\tCompiling FFNN Theano functions...')

    # retrieve all parameters from the network
    all_params = lg.layers.get_all_params(output_layer, trainable=True)

    # scale learning rate by a factor of 0.9 if momentum is applied,
    # to counteract the larger update steps that momentum yields
    learning_rate = exp.learning_rate - 0.9 * exp.learning_rate * exp.momentum

    # define updates and adapt if momentum is applied (*_out[1] is the cost)
    updates = lg.updates.adagrad(loss_or_grads=stochastic_out[1],
                                 params=all_params,
                                 learning_rate=learning_rate)
    if exp.momentum:
        updates = lg.updates.apply_nesterov_momentum(updates)

    # compute stochastic cost and output, and update params
    train_model = theano.function(inputs=[input_layer.input_var, target],
                                  outputs=stochastic_out,
                                  updates=updates)

    # compute deterministic cost and output, and don't update
    test_model = theano.function(inputs=[input_layer.input_var, target],
                                 outputs=deterministic_out)

    # compute deterministic output, don't update
    output = lg.layers.get_output(output_layer, deterministic=True)
    predict_model = theano.function(inputs=[input_layer.input_var],
                                    outputs=output)

    return train_model, test_model, predict_model


def iter_minibatches(X, Y, batch_size):
    """ Iterate over rows in X, Y in mini-batches. """

    assert X.shape[0] == Y.shape[0]

    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield X[excerpt], Y[excerpt]


def fit(model, train_input, train_target, out_dir):
    """
    Fit the hybrid classifier to a training dataset of song-features
    and song-playlist examples. Return nothing.

    Parameters
    ----------
    model: model file
        Model specification.
    train_input: numpy array, shape (num songs, feature size)
        Input array of song features.
    train_target: numpy array, shape (num songs, num playlists)
        Target array of playlists the songs belong to.
    out_dir: string
        Path to the results directory
    """

    print('\nSetting up fit...')

    # identify dimensions
    feat_size = train_input.shape[1]
    n_classes = train_target.shape[1]

    # build network
    input_layer, output_layer = build_model(feat_size, n_classes, model)

    # define theano variables
    theano_vars = declare_theano_variables(output_layer, model)
    target, stochastic_metrics, deterministic_metrics = theano_vars

    # define theano functions
    train_model, _, _ = compile_theano_functions(
        input_layer, output_layer, target, stochastic_metrics,
        deterministic_metrics, model
    )

    # fit the classifier
    print('\nFitting...')

    for epoch in xrange(1, model.max_epochs + 1):

        # keep track of time
        start_time = time.time()

        # full pass over the training data
        output_list = []
        train_cost, num_batches = 0, 0

        for batch in iter_minibatches(train_input, train_target,
                                      model.batch_size):
            b_input, b_target = batch
            b_output, b_cost = train_model(b_input, b_target.toarray())
            train_cost += float(b_cost)  # theano function returns an array
            num_batches += 1
            output_list.append(b_output)

        # shuffle training data before the next pass
        train_input, train_target = shuffle(train_input, train_target)

        # inform about the elapsed time
        print('\tEpoch {} of {} took {:.3f}s'.format(
            epoch, model.max_epochs, time.time() - start_time)
        )

    # save the fit model
    print('\nSaving model weights...')

    params = lg.layers.get_all_param_values(output_layer)
    params_file = '{}_params.pkl'.format(model.name)
    with open(os.path.join(out_dir, params_file), 'w') as f:
        cPickle.dump(params, f)


def test(model, test_input, test_target, train_target, out_dir, song_obs=None):
    """
    Evaluate the playlist continuations given by a fit hybrid classifier.

    Parameters
    ----------
    model: model file
        Model specification.
    test_input: numpy array, shape (num songs, feature size)
        Input array of song features.
    test_target: numpy array, shape (num songs, num playlists)
        Matrix of song-playlist co-occurrences at the test split.
    train_target: numpy array, shape (num_songs, num playlists)
        Matrix of song-playlist co-occurrences at the train split.
    out_dir: string
        Path to the results directory
    song_obs: int
        Test on songs observed song_obs times during training.
    """

    print('\nSetting up test...')

    # identify dimensions
    feat_size = test_input.shape[1]
    n_classes = test_target.shape[1]

    # build network
    input_layer, output_layer = build_model(feat_size, n_classes, model)

    # define theano variables
    theano_vars = declare_theano_variables(output_layer, model)
    target, stochastic_metrics, deterministic_metrics = theano_vars

    # define theano functions
    _, _, predict_model = compile_theano_functions(
        input_layer, output_layer, target, stochastic_metrics,
        deterministic_metrics, model
    )

    # load previously fit hybrid classifier weights
    print('\nLoading fit weights to the model...')

    params_file = '{}_params.pkl'.format(model.name)
    if os.path.isfile(os.path.join(out_dir, params_file)):
        with open(os.path.join(out_dir, params_file), 'rb') as f:
            params = cPickle.load(f)
    else:
        sys.exit('\tThe file {} does not exist yet. You need to fit the model '
                 'first.'.format(os.path.join(out_dir, params_file)))

    # load the weights on the defined model
    lg.layers.set_all_param_values(output_layer, params)

    # use the classifier to populate a matrix of song-playlist scores
    # (we will then use its transpose)
    print('\nPredicting playlist-song scores...')
    test_output = predict_model(test_input)

    # use the scores to extend query playlists
    print('\nEvaluating playlist continuations...')

    # mask known good continuations from training
    mask_training_targets(test_output, train_target, verbose=True)

    # keep only songs with song_obs observations at training time
    if song_obs is not None:
        occ = np.array(train_target.sum(axis=1)).flatten()
        test_target[np.where(occ != song_obs)[0], :] = 0
        test_target.eliminate_zeros()

    # find rank of actual continuations
    song_rank = find_rank(test_output.T, test_target.T.tocsr(), verbose=False)

    # compute mean average precision
    song_avgp = compute_map(test_output.T, test_target.T.tocsr(), verbose=False)

    # compute precision recall
    song_prc, song_rec = compute_precision_recall(
        test_output.T, test_target.T.tocsr(), [10, 30, 100], verbose=False)

    # report metrics
    song_metrics = [np.median(song_rank), np.mean(song_avgp)]
    for K in [10, 30, 100]:
        song_metrics += [np.mean(song_rec[K])]

    metrics = ['med_rank', 'map', 'mean_rec10', 'mean_rec30', 'mean_rec100']
    print(('\n\t' + '{:<13}' * 5).format(*metrics))
    print(('\t' + '{:<13.1f}' * 1 + '{:<13.2%}' * 4).format(*song_metrics))
