# Model specification file

from lasagne_wrapper.training_strategy import TrainingStrategy, RefinementStrategy
from lasagne_wrapper.optimization_objectives import mean_pixel_binary_crossentropy
from lasagne_wrapper.parameter_updates import get_update_adam
from lasagne_wrapper.learn_rate_shedules import get_constant
from lasagne_wrapper.batch_iterators import BatchIterator

import theano.tensor as T
import theano
import lasagne as lg
import numpy as np


# dp settings
mode = 'fix'
feature = 'ivectors'
standardize = False
normalize = None

# model settings
FEAT_DIM = 200
SAMPLE_SIZE = 50
BATCH_SIZE = 2 * SAMPLE_SIZE

N_HIDDEN = 512
L2 = 0.0001


def create_model(batch_size, pl_length, feat_dim):
    """ Compile matching network """

    # input layers
    l_in_pl = lg.layers.InputLayer(shape=(batch_size, pl_length, feat_dim))
    l_in_sg = lg.layers.InputLayer(shape=(batch_size, feat_dim))

    # reshape playlist to song feature level -> bs * pl_length, feat_dim
    net_pl = lg.layers.ReshapeLayer(l_in_pl, shape=(-1, feat_dim))

    # concatenate playlist songs with target songs
    shared_feat_net = lg.layers.ConcatLayer([net_pl, l_in_sg], axis=0)

    # feature processing network
    for i in xrange(2):
        shared_feat_net = lg.layers.DenseLayer(shared_feat_net, num_units=N_HIDDEN)
        shared_feat_net = lg.layers.batch_norm(shared_feat_net)
        shared_feat_net = lg.layers.dropout(shared_feat_net, p=0.25)

    # split playlists and target songs again
    split_idx = batch_size * pl_length
    net_pl = lg.layers.SliceLayer(shared_feat_net, indices=slice(0, split_idx), axis=0)
    net_sg = lg.layers.SliceLayer(shared_feat_net, indices=slice(split_idx, None), axis=0)

    # reshape playlist to (batch_size, pl_length, hidden_units) again
    net_pl = lg.layers.ReshapeLayer(net_pl, shape=(batch_size, pl_length, net_pl.output_shape[-1]))

    # average along song axis
    net_pl = lg.layers.ExpressionLayer(net_pl, lambda X: X.mean(1), output_shape='auto')

    # merge two streams again but this time in feature direction
    net = lg.layers.ConcatLayer((net_pl, net_sg), axis=1)

    # decision  network
    for _ in xrange(2):
        net = lg.layers.DenseLayer(net, num_units=N_HIDDEN)
        net = lg.layers.batch_norm(net)
        net = lg.layers.dropout(net, p=0.5)

    # binary decision
    net = lg.layers.DenseLayer(net, num_units=1, nonlinearity=lg.nonlinearities.sigmoid)

    return net


def prepare(group_mb, mask_mb, positive_mb, negative_mb):
    """
    Prepare batch data before being passed to the network.

    Parameters
    ----------
    group_mb: numpy array of shape (num samples, max(len(group)), feature_size)
        Each row contains the feature vectors of the items within a
        sample's group. They are stored in depth.
    mask_mb: dummy argument to keep the interface
    positive_mb: numpy array of shape (num samples, feature_size)
        Each row contains the feature vector of a sample's positive item.
    negative_mb: numpy array of shape (num samples, feature_size)
        Each row contains the feature vector of a sample's negative item.

    Returns
    -------
     group_mb: numpy array, shape (2*num samples, max(len(group)), feature_size)
        Stack the input group_mb.
    item_mb: numpy array of shape (2 * num samples, feature_size)
        Stack positive_mb and negative_mb.
    label_mb: numpy array of shape (2 * num samples, )
        Stack ones and zeros, indicating positive and negative items.
    """

    group_mb = np.vstack((group_mb, group_mb))
    item_mb = np.vstack((positive_mb, negative_mb))
    label_mb = np.hstack((np.ones(len(positive_mb)), np.zeros(len(negative_mb))))

    minibatch = [
        group_mb.astype(theano.config.floatX),
        item_mb.astype(theano.config.floatX),
        label_mb.astype(theano.config.floatX)
    ]

    return minibatch


def get_batch_iterator():
    """
    Get batch iterator
    """

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


refinement_strategy = RefinementStrategy(n_refinement_steps=5, refinement_patience=5, learn_rate_multiplier=0.5)


train_strategy = TrainingStrategy(
    batch_size=SAMPLE_SIZE,
    ini_learning_rate=0.001,
    max_epochs=1000,
    patience=5,
    L2=None,
    samples_per_epoch=None,
    refinement_strategy=refinement_strategy,
    y_tensor_type=T.vector,
    objective=mean_pixel_binary_crossentropy,
    adapt_learn_rate=get_constant(),
    update_function=get_update_adam(),
    train_batch_iter=get_batch_iterator(),
    valid_batch_iter=get_batch_iterator()
)
