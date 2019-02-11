# Song-to-playlist classifier utils.

from __future__ import print_function
from __future__ import division

from utils.evaluation import compute_metrics, summarize_metrics

from sklearn.utils import check_random_state, shuffle
from tqdm import tqdm

import theano.tensor as T
import theano
import lasagne as lg
import numpy as np

import cPickle
import time
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
        '\tEarly-stop options\n'
        '\tpatience = {}\n'
        '\trefinement = {}\n'
        '\tfactor_lr = {}\n'
        '\tmax_epochs_increase = {}\n'
        '\tsignificance_level = {}\n\n'
        '\tRegularization\n'
        '\tinput_dropout = {}\n'
        '\thidden_dropout = {}\n'
        '\tpositive_weight = {}\n'
        '\tnonpositive_weight = {}\n'
        '\tl1_weight = {}\n'
        '\tl2_weight = {}\n\n'
        '\tFeatures\n'
        '\tfeature = {}\n'
        '\tstandardize = {}\n'
        '\tnormalize = {}'.format(
            model.n_layers, model.n_hidden, model.hid_nl, model.out_nl,
            model.batch_size, model.learning_rate, model.max_epochs,
            model.momentum, model.patience, model.refinement, model.factor_lr,
            model.max_epochs_increase, model.significance_level,
            model.input_dropout, model.hidden_dropout, model.positive_weight,
            model.nonpositive_weight, model.l1_weight, model.l2_weight,
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
        print('\tBuilding model...', end='')

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
    if model.l1_weight > 0:
        l1_reg = lg.regularization.regularize_network_params(output_layer, lg.regularization.l1)
        cost += model.l1_weight * l1_reg

    if model.l2_weight > 0:
        l2_reg = lg.regularization.regularize_network_params(output_layer, lg.regularization.l2)
        cost += model.l2_weight * l2_reg

    return output, cost


def declare_theano_variables(output_layer, model, verbose=True):
    """
    Define target, network output, cost and learning rate.

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
    learning_rate: Theano shared variable
        Learning rate for the optimizers.
    """

    if verbose:
        print('\tDeclaring theano variables...')

    # scale learning rate by a factor of 0.9 if momentum is applied,
    # to counteract the larger update steps that momentum yields
    lr = model.learning_rate - 0.9 * model.learning_rate * model.momentum
    learning_rate = theano.shared(np.asarray(lr, dtype=theano.config.floatX))

    # define target placeholder for the cost functions
    target = T.bmatrix('target')

    # stochastic cost expression
    stochastic_out = define_cost(output_layer, target, model, determ=False)

    # deterministic cost expression
    deterministic_out = define_cost(output_layer, target, model, determ=True)

    return target, stochastic_out, deterministic_out, learning_rate


def compile_theano_functions(input_layer, output_layer, target, stochastic_out,
                             deterministic_out, learning_rate, model, verbose=True):
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
    learning_rate: Theano shared variable
        Learning rate for the optimizers.
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
        print('\tCompiling theano functions...')

    # retrieve all parameters from the network
    all_params = lg.layers.get_all_params(output_layer, trainable=True)

    # define updates and adapt if momentum is applied (*_out[1] is the cost)
    updates = lg.updates.adagrad(loss_or_grads=stochastic_out[1],
                                 params=all_params,
                                 learning_rate=learning_rate)
    if model.momentum:
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

    # do as many minibatches of batch_size as possible
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield X[excerpt], Y[excerpt]

    # do a final small minibatch if some samples remain
    if X.shape[0] % batch_size != 0:
        last_start = int(np.floor(X.shape[0] / batch_size)) * batch_size
        excerpt = slice(last_start, None)
        yield X[excerpt], Y[excerpt]


def train(model, train_input, train_target, valid_input, valid_target, out_dir,
          random_state):
    """
    Train the hybrid classifier to a training dataset of song-features and
    song-playlist examples. Monitoring on a validation dataset. Return nothing.

    Parameters
    ----------
    model: model file
        Model specification.
    train_input: numpy array, shape (num songs, feature size)
        Input array of song features for training.
    train_target: numpy array, shape (num songs, num playlists)
        Target array of playlists the songs belong to for training.
    valid_input: numpy array, shape (num songs, feature size)
        Input array of song features for validation.
    valid_target: numpy array, shape (num songs, num playlists)
        Target array of playlists the songs belong to for validation.
    out_dir: string
        Path to the params and logging directory
    random_state: None, int or numpy RandomState
        Used to shuffle.
    """

    # set random behavior
    rng = check_random_state(random_state)

    print('\nSetting up training...')

    # identify dimensions
    feat_size = train_input.shape[1]
    n_classes = train_target.shape[1]

    # build network
    input_layer, output_layer = build_model(feat_size, n_classes, model)

    # define theano variables
    theano_vars = declare_theano_variables(output_layer, model)
    target, stochastic_metrics, deterministic_metrics, learning_rate = theano_vars

    # define theano functions
    train_model, test_model, predict_model = compile_theano_functions(
            input_layer, output_layer, target, stochastic_metrics,
            deterministic_metrics, learning_rate, model
    )

    # set up metrics monitoring
    metrics = ['cost', 'med_rank', 'mrr', 'map', 'mean_rec10', 'mean_rec30', 'mean_rec100']
    train_log = {metric: [] for metric in metrics}
    valid_log = {metric: [] for metric in metrics}
    file_log = '{}_log_train.pkl'.format(model.name)

    # initialize best epoch info
    best_valid_cost = np.inf
    best_epoch = 1
    best_params = lg.layers.get_all_param_values(output_layer)
    best_file = '{}_best.pkl'.format(model.name)
    with open(os.path.join(out_dir, best_file), 'wb') as f:
        cPickle.dump((best_valid_cost, best_epoch, best_params), f)

    # initialize early stop and learning rate schedule
    early_stop = False
    epoch = 1
    max_epochs = model.max_epochs
    patience = model.patience
    refinement = model.refinement

    # train the classifier
    print('\nTraining...')

    while epoch <= max_epochs and not early_stop:

        # keep track of time
        start_time = time.time()

        # shuffle training data before each pass
        train_input, train_target = shuffle(train_input, train_target, random_state=rng)

        # training on mini-batches

        train_cost = 0.
        num_batches = 0

        if epoch % EVERY != 0:

            # do not compute ranking metrics

            for batch in iter_minibatches(train_input, train_target, model.batch_size):
                batch_input, batch_target = batch
                _, batch_cost = train_model(batch_input, batch_target.toarray())
                train_cost += np.asscalar(batch_cost)  # theano returns an array
                num_batches += 1

            # put together batches
            train_log['cost'].append(train_cost / num_batches)

        else:

            # compute ranking metrics

            output_list = []
            for batch in iter_minibatches(train_input, train_target, model.batch_size):
                batch_input, batch_target = batch
                batch_output, batch_cost = train_model(batch_input, batch_target.toarray())
                train_cost += np.asscalar(batch_cost)  # theano returns an array
                num_batches += 1
                output_list.append(batch_output)

            # put together batches
            train_log['cost'].append(train_cost / num_batches)
            train_output = np.vstack(output_list)

            # compute training metrics (transpose to have playlists as rows)
            train_metrics = compute_metrics(train_output.T, train_target.T.tocsr(), k_list=[10, 30, 100], verbose=False)
            train_metrics = summarize_metrics(*train_metrics, k_list=[10, 30, 100], ci=False, pivotal=False, verbose=False)

        # validation on single batch

        valid_output, valid_cost = test_model(valid_input, valid_target.toarray())
        valid_cost = np.asscalar(valid_cost)  # theano returns an array
        valid_log['cost'].append(valid_cost)

        if epoch % EVERY == 0:

            # compute validation metrics (transpose to have playlists as rows)
            valid_metrics = compute_metrics(valid_output.T, valid_target.T.tocsr(), k_list=[10, 30, 100], verbose=False)
            valid_metrics = summarize_metrics(*valid_metrics, k_list=[10, 30, 100], ci=False, pivotal=False, verbose=False)

            print(('\n\t\t' + '{:<13}' + '{:<13}' * 6).format('split', *metrics[1:]))
            print(('\t\t' + '{:<13}' + '{:<13.1f}' * 1 + '{:<13.2%}' * 5).format('train', *train_metrics))
            print(('\t\t' + '{:<13}' + '{:<13.1f}' * 1 + '{:<13.2%}' * 5).format('valid', *valid_metrics))
            print('')

            for m, tm, vm in zip(metrics[1:], train_metrics, valid_metrics):
                train_log[m].append(tm)
                valid_log[m].append(vm)

        print('\tEpoch {} of {} took {:.3f}s'.format(epoch, max_epochs, time.time() - start_time))

        # revisit best epoch details

        if valid_cost < best_valid_cost:

            if valid_cost < best_valid_cost * model.significance_level:

                # extend max_epochs if the improvement is significant
                if max_epochs < int(epoch * model.max_epochs_increase):
                    max_epochs = int(epoch * model.max_epochs_increase)
                    print('\n\tSet max_epochs to {}.\n'.format(max_epochs))

            # update best setting
            best_valid_cost = valid_cost
            best_epoch = epoch
            best_params = lg.layers.get_all_param_values(output_layer)

        else:

            # decrease patience
            patience -= 1
            print('\n\tDecrease patience. Currently patience={}, refinement={}.'.format(patience, refinement))

            if patience == 0:
                print('\n\tPatience exhausted: restoring best model...')
                lg.layers.set_all_param_values(output_layer, best_params)

                if refinement > 0:

                    # decrease refinement
                    refinement -= 1
                    print('\n\tDecrease refinement. Currently patience={}, refinement={}.'.format(patience, refinement))

                    # update learning rate
                    old_lr = learning_rate.get_value()
                    new_lr = np.asarray(old_lr * model.factor_lr, dtype=theano.config.floatX)
                    learning_rate.set_value(new_lr)
                    print('\n\tUpdate learning rate to {}.'.format(new_lr))

                    # restore patience
                    patience = model.patience
                    print('\n\tRestore patience. Currently patience={}, refinement={}.'.format(patience, refinement))

                else:
                    print('\n\tPatience and refinement steps exhausted. '
                          'Early stopping!')
                    early_stop = True

            elif epoch == max_epochs:
                print('\n\tReached max_epochs without improvement.')

        epoch += 1

    print('\nBest valid cost was {:.6f} at epoch {}.'.format(best_valid_cost, best_epoch))

    # save metrics and best setting
    with open(os.path.join(out_dir, file_log), 'wb') as f:
        cPickle.dump((train_log, valid_log), f)

    with open(os.path.join(out_dir, best_file), 'wb') as f:
        cPickle.dump((best_valid_cost, best_epoch, best_params), f)


def fit(model, fit_input, fit_target, out_dir, random_state):
    """
    Fit the hybrid classifier to a training dataset of song-features and
    song-playlist examples. Return nothing.

    Parameters
    ----------
    model: model file
        Model specification.
    fit_input: numpy array, shape (num songs, feature size)
        Input array of song features.
    fit_target: numpy array, shape (num songs, num playlists)
        Target array of playlists the songs belong to.
    out_dir: string
        Path to the params and logging directory
    random_state: None, int or numpy RandomState
        Used to shuffle.
    """

    # set random behavior
    rng = check_random_state(random_state)

    print('\nSetting up fit...')

    # identify dimensions
    feat_size = fit_input.shape[1]
    n_classes = fit_target.shape[1]

    # build network
    input_layer, output_layer = build_model(feat_size, n_classes, model)

    # define theano variables
    theano_vars = declare_theano_variables(output_layer, model)
    target, stochastic_metrics, deterministic_metrics, learning_rate = theano_vars

    # define theano functions
    train_model, _, _ = compile_theano_functions(
        input_layer, output_layer, target, stochastic_metrics,
        deterministic_metrics, learning_rate, model
    )

    # set up metrics monitoring and params file
    metrics = ['cost', 'med_rank', 'mrr', 'map', 'mean_rec10', 'mean_rec30', 'mean_rec100']
    log = {metric: [] for metric in metrics}
    log_file = '{}_log_fit.pkl'.format(model.name)
    params_file = '{}_params.pkl'.format(model.name)

    # fit the classifier
    print('\nFitting...')
    start = time.time()

    for epoch in tqdm(xrange(1, model.max_epochs + 1)):

        # shuffle training data before every pass
        fit_input, fit_target = shuffle(fit_input, fit_target, random_state=rng)

        # fitting on mini-batches

        fit_cost = 0.
        num_batches = 0

        if epoch % EVERY != 0:

            # do not compute ranking metrics

            for batch in iter_minibatches(fit_input, fit_target, model.batch_size):
                b_input, b_target = batch
                _, b_cost = train_model(b_input, b_target.toarray())
                fit_cost += np.asscalar(b_cost)  # theano returns an array
                num_batches += 1

            # put together batches
            log['cost'].append(fit_cost / num_batches)

        else:

            # compute ranking metrics

            output_list = []

            for batch in iter_minibatches(fit_input, fit_target, model.batch_size):
                b_input, b_target = batch
                b_output, b_cost = train_model(b_input, b_target.toarray())
                fit_cost += np.asscalar(b_cost)  # theano returns an array
                num_batches += 1
                output_list.append(b_output)

            # put together batches
            log['cost'].append(fit_cost / num_batches)
            fit_output = np.vstack(output_list)

            # compute training metrics (transpose to have playlists as rows)
            fit_metrics = compute_metrics(fit_output.T, fit_target.T.tocsr(), k_list=[10, 30, 100], verbose=False)
            fit_metrics = summarize_metrics(*fit_metrics, k_list=[10, 30, 100], ci=False, pivotal=False, verbose=False)

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
    params = lg.layers.get_all_param_values(output_layer)
    with open(os.path.join(out_dir, params_file), 'w') as f:
        cPickle.dump(params, f)


def compute_scores(model, params_dir, cont_input, cont_target):
    """
    Compute the song-playlist scores.

    Parameters
    ----------
    model: model file
        Model specification.
    params_dir: string
        Path to the directory with previously fit parameters.
    cont_input: numpy array, shape (num songs, feature size)
        Input array of song features.
    cont_target: numpy array, shape (num songs, num playlists)
        Matrix of song-playlist co-occurrences at the continuation split.
    """

    # identify dimensions
    feat_size = cont_input.shape[1]
    n_classes = cont_target.shape[1]

    # build network
    input_layer, output_layer = build_model(feat_size, n_classes, model)

    # define theano variables
    theano_vars = declare_theano_variables(output_layer, model)
    target, stochastic_metrics, deterministic_metrics, learning_rate = theano_vars

    # define theano functions
    _, _, predict_model = compile_theano_functions(
        input_layer, output_layer, target, stochastic_metrics,
        deterministic_metrics, learning_rate, model
    )

    # load previously fit hybrid classifier weights
    print('\nLoading fit weights to the model...')

    params_file = '{}_params.pkl'.format(model.name)
    if os.path.isfile(os.path.join(params_dir, params_file)):
        with open(os.path.join(params_dir, params_file), 'rb') as f:
            params = cPickle.load(f)
    else:
        sys.exit('\tThe file {} does not exist yet. You need to fit the model '
                 'first.'.format(os.path.join(params_dir, params_file)))

    # load the weights on the defined model
    lg.layers.set_all_param_values(output_layer, params)

    # use the classifier to populate a matrix of song-playlist scores
    print('\nPredicting song-playlist scores...')
    start = time.time()
    cont_output = predict_model(cont_input)
    print('\nTime predicting: {} sec.'.format(round(time.time() - start, 4)))

    return cont_output
