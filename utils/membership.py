# Membership utils and data pool class.

from __future__ import print_function
from __future__ import division

from sklearn.utils import check_random_state
from sklearn import preprocessing
from scipy import misc
from tqdm import tqdm

import theano
import numpy as np
import itertools as it

import os


class DataPool(object):
    """
    The DataPool class deals with groups of items. Given a group, each item
    within is assumed to be a positive item and items outside are assumed to be
    negative items. Given a list of groups and feature vectors describing the
    items, each data sample is a triplet (group, positive item, negative item)
    of arrays, each containing the corresponding item features.

    Parameters
    ----------
    groups: list of lists
        Each item is a group represented as a list of item ids.
    filter_size: int
        Minimum group size required. Do nothing if filter_size = 0.
    mode: str
        Either 'fix' or 'variable'. If 'fix', all the groups are required to
        have the same size. If 'variable', each group can have its own size.
    out_size: int
        Size of the prepared groups that the DataPool outputs. This is mostly
        relevant for mode='fix', but technically in mode='variable' the groups
        also have the same output size, only that a mask indicates which
        positions are empty. In mode='fix' this value can not be larger than
        the smallest group size in the data because we don't accept repeated
        items in a group, and if out_size = 0, the smallest group size is taken
        as the output size. In mode='variable' this value can not be smaller
        than the largest group size, and if out_size = 0, the largest group
        size is taken as the output size.
    sampling_factor: int
        If mode='fix', sample no more than sampling_factor * len(group)
        sub-groups out of each group.
    features: dict
        Song features indexed by MSD song id.
    standardize: bool
        Standardize features if True.
    scaler: None or sklearn.preprocessing.data.{RobustScaler, StandardScaler}
        If standardize is True, define an own 'scaler' or take one trained on
        other data. The latter is useful to standardize test data using a
        scaler fit on training data.
    normalize: None or str
        Normalize if not None. In this case, either 'l1' or 'l2', indicating
        under which norm the features are normalized.
    random_state: None, int or numpy RandomState
        Sets the pseudo-random behaviour.
    """

    def __init__(self, groups=None, filter_size=0, mode='fix', out_size=0,
                 sampling_factor=1, features=None, standardize=False,
                 scaler=None, normalize=None, random_state=None):

        self.groups = groups
        self.filter_size = filter_size
        self.mode = mode
        self.out_size = out_size
        self.sampling_factor = sampling_factor
        self.features = features.copy()
        self.standardize = standardize
        self.scaler = scaler
        self.normalize = normalize
        self.rng = check_random_state(random_state)

        self.original_groups = None
        self.num_groups = None
        self.min_size = None
        self.max_size = None
        self.positive_pairs = None
        self.num_positive_pairs = None
        self.shape = None
        self.items = set([item for group in groups for item in group])
        self.num_items = len(self.items)
        self.feature_size = self.features.itervalues().next().shape[0]

        self._prepare_groups()
        self._prepare_positive_pairs()
        self.shuffle()
        self._prune_features()
        if standardize or normalize is not None:
            self._preprocess_features()

    def _prepare_groups(self):
        """
        Prepare groups. Filter out groups smaller than filter_size.
        In mode='fix', draw sub-groups of size `out_size`. Groups of size
        `out_size` remain as they are. For larger groups we randomly draw no
        more than sampling_factor * len(group) sub-groups of size `out_size`
        without replacement. In mode='variable', leave the groups as they are.
        Always keep a copy of the original group a sub-group belongs to. This
        is required to later draw the negative items correctly.
        """

        # filter out small groups
        if self.filter_size > 0:
            self.groups = [g for g in self.groups if len(g) >= self.filter_size]

            if len(self.groups) == 0:
                raise AssertionError('No groups remain after filtering. '
                                     'Consider using a smaller min_size.')

        # minimum and maximum group size after filtering
        self.min_size = min([len(g) for g in self.groups])
        self.max_size = max([len(g) for g in self.groups])

        if self.mode == 'variable':

            # assert out_size >= max_size
            if (self.out_size > 0) and (self.out_size < self.max_size):
                raise AssertionError(
                    'out_size < max_size ({} < {}). This is not allowed in '
                    'mode=\'variable\'.'.format(self.out_size, self.max_size)
                )

            # set out_size to the largest group size if out_size is 0
            if self.out_size == 0:
                self.out_size = self.max_size

            # leave the groups as they are
            original_groups = self.groups
            prepared_groups = self.groups

        else:

            # assert out_size <= min_size because we sample without replacement
            if self.out_size > self.min_size:
                raise AssertionError(
                    'out_size > min_size ({} > {}). In mode=\'fix\' this is '
                    'not allowed because we sample without replacement.'.format(
                        self.out_size, self.min_size)
                )

            # set out_size to the minimum group size if out_size is 0
            if self.out_size == 0:
                self.out_size = self.min_size

            # prepare fix-size groups by sampling sub-groups out of each group
            original_groups, prepared_groups = [], []
            for g in self.groups:

                # number of possible draws from group
                num_draws = misc.comb(len(g), self.out_size, exact=True)

                # limit the number of draws if there are too many
                if num_draws > self.sampling_factor * len(g):
                    num_draws = self.sampling_factor * len(g)
                    draws = [list(self.rng.choice(a=g, size=self.out_size, replace=False)) for _ in xrange(num_draws)]
                else:
                    draws = [list(draw) for draw in it.combinations(g, self.out_size)]

                original_groups += [g] * num_draws
                prepared_groups += draws

        # check that we keep an original group for each prepared group
        assert len(original_groups) == len(prepared_groups)

        self.original_groups = original_groups
        self.groups = prepared_groups
        self.num_groups = len(prepared_groups)

    def _prepare_positive_pairs(self):
        """
        Pair each group with each of its items. Groups are represented by their
        index in `groups` and items by their index within each group.
        Also get the number of resulting positive pairs.
        """

        positive_pairs = []
        for group_idx in xrange(self.num_groups):
            for item_idx in xrange(len(self.groups[group_idx])):
                positive_pair = np.array([group_idx, item_idx], dtype=np.int)
                positive_pairs.append(positive_pair)

        self.positive_pairs = np.vstack(positive_pairs)
        self.num_positive_pairs = self.positive_pairs.shape[0]
        self.shape = [self.num_positive_pairs]

    def shuffle(self):
        """ Shuffle positive_pairs. """

        self.rng.shuffle(self.positive_pairs)

    def _prune_features(self):
        """ Keep only features corresponding to items in the DataPool. """

        for item in self.features.keys():
            if item not in self.items:
                del self.features[item]

    def _preprocess_features(self):
        """ Standardize and normalize features. """

        if self.standardize:

            if self.scaler is None:
                # define a new scaler if none is given
                self.scaler = preprocessing.RobustScaler()

                # store all features into numpy array and fit standardizer
                X = np.vstack([self.features[item].reshape(1, -1) for item in self.items])
                self.scaler.fit(X)

        for item in self.features:
            x = self.features[item].reshape(1, -1)

            if self.standardize:
                x = self.scaler.transform(x)

            if self.normalize is not None:
                x = preprocessing.normalize(x, norm=self.normalize)

            self.features[item] = x.flatten()

    def _shape_sample(self, pair_idx):
        """
        Given a <group, positive item> pair, find a negative item. Furthermore,
        build the array of features for group, positive item and negative item.
        Such a triplet constitutes a data sample. The group features need to be
        right-padded with zeros if the group is smaller than out_size because
        this way we can stack samples into minibatches. Therefore we also build
        a mask indicating empty item positions in the group.

        Parameters
        ----------
        pair_idx: int
            Index in positive_pairs yielding a <group, positive item> pair.

        Returns
        -------
        group_features: numpy array of shape (1, out_size, feature_size)
            Feature vectors of items in group stored in depth. Right-pad with
            zeros if the group is smaller than out_size.
        group_mask: numpy array of shape (1, out_size)
            Binary array indicating empty item positions in the group.
        positive_features: numpy array of shape (1, feature_size)
            Feature vector of the positive_item.
        negative_features: numpy array of shape (1, feature_size)
            Feature vector of the negative_item.
        """

        # group and positive item idx
        group_idx, item_idx = self.positive_pairs[pair_idx]
        group = list(self.groups[group_idx])  # list copies to remove item later

        # draw negative item discarding the items from the original group
        original_group = self.original_groups[group_idx]
        candidates = self.items.difference(set(original_group))
        negative_item = self.rng.choice(list(candidates))

        # positive item and remove from group
        positive_item = group.pop(item_idx)

        # get the features for the items in the group (store in depth dim)
        group_features = np.hstack([self.features[item].reshape(1, 1, -1) for item in group])

        # build group mask
        group_mask = np.ones(shape=(1, len(group)))

        # right-pad if group is too small
        if len(group) < self.out_size:
            pad_width = self.out_size - len(group) - 1

            # pad features
            zeros_features = np.zeros(shape=(1, pad_width, self.feature_size))
            group_features = np.hstack([group_features, zeros_features])

            # pad mask
            zeros_mask = np.zeros(shape=(1, pad_width))
            group_mask = np.hstack([group_mask, zeros_mask])

        # get features for positive_item and negative_item
        positive_features = self.features[positive_item].reshape(1, -1)
        negative_features = self.features[negative_item].reshape(1, -1)

        return group_features, group_mask, positive_features, negative_features

    def __getitem__(self, index):
        """
        Make (group, positive item, negative item) samples be accessible by
        an integer index or a slice index. In both cases, return numpy arrays
        where each row corresponds to one sample.

        Parameters
        ----------
        index: int or slice
            Access a single data element or a slice.

        Returns
        -------
        group_mb: numpy array of shape (len(index), out_size, feature_size)
            Each row contains the feature vectors of the items within a
            sample's group. They are stored in depth. When a group is smaller
            than out_size, it is right-padded with zeros.
        mask_mb: numpy array of shape (len(index), out_size)
            Each row contains the mask of a sample's group indicating the
            right-padded positions if the group is too small.
        positive_mb: numpy array of shape (len(index), feature_size)
            Each row contains the feature vector of a sample's positive item.
        negative_mb: numpy array of shape (len(index), feature_size)
            Each row contains the feature vector of a sample's negative item.
        """

        # collect the data
        if index.__class__ == slice:

            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.num_positive_pairs
            group_mb, mask_mb, positive_mb, negative_mb = [], [], [], []

            for batch_idx, pair_idx in enumerate(range(start, stop)):
                group, mask, positive, negative = self._shape_sample(pair_idx)
                group_mb.append(group)
                mask_mb.append(mask)
                positive_mb.append(positive)
                negative_mb.append(negative)

            group_mb = np.vstack(group_mb)
            mask_mb = np.vstack(mask_mb)
            positive_mb = np.vstack(positive_mb)
            negative_mb = np.vstack(negative_mb)

        else:
            group_mb, mask_mb, positive_mb, negative_mb = self._shape_sample(index)

        minibatch = [
            group_mb.astype(theano.config.floatX),
            mask_mb.astype(theano.config.floatX),
            positive_mb.astype(theano.config.floatX),
            negative_mb.astype(theano.config.floatX)
        ]

        return minibatch


def tolist(playlists_idx, songs_idx, position, idx2song, subset=None,
           verbose=True):
    """
    Convert playlists in coordinates to list of lists.

    Parameters
    ----------
    playlists_idx: numpy array, shape (nnz, )
        Each element is a playlist coordinate.
    songs_idx: numpy array, shape (nnz, )
        Each element is a song coordinate.
    position: numpy array, shape (nnz, )
        Position of the corresponding song in the corresponding playlist.
    idx2song: dict
        Mapping between the song idx and the MSD song id.
    subset: numpy array, shape (split size, )
        Subsets the training playlists_idx, songs_idx and position coordinates.
    verbose: bool
        Print info if True.

    Returns
    -------
    playlists: list
        Each element is a playlist.
    """

    if verbose:
        print('\nConverting playlists from coordinates to list of lists...')

    if subset is None:
        subset = range(len(playlists_idx))

    # make sure that the coordinates are sorted by playlist_idx and position
    coo = zip(playlists_idx[subset], songs_idx[subset], position[subset])
    coo.sort(key=lambda triplet: (triplet[0], triplet[2]))

    # format in lists
    playlists = []
    current_playlist = []
    current_pl_idx = min(playlists_idx[subset])

    for pl_idx, sg_idx, _ in coo:

        if pl_idx == current_pl_idx:
            current_playlist.append(idx2song[sg_idx])

        else:
            playlists.append(current_playlist)
            current_playlist = [idx2song[sg_idx]]
            current_pl_idx = pl_idx

    playlists.append(current_playlist)

    return playlists


def print_datapool(dp, split):
    """ Print dataset details """

    print('')
    print('\t{} set:'.format(split))
    print('\t\tnum playlists = {}'.format(dp.num_groups))
    print('\t\tmin playlist length = {}'.format(dp.min_size))
    print('\t\tmax playlist length = {}'.format(dp.max_size))
    print('\t\tout playlist length = {}'.format(dp.out_size))
    print('\t\tnum songs = {}'.format(dp.num_items))
    print('\t\tnum positive pairs = {}'.format(dp.num_positive_pairs))
    print('\t\tstandardize features = {}'.format(dp.standardize))
    print('\t\tnormalize features = {}'.format(dp.normalize))


def shape_datapools(playlists_idx, songs_idx, position, features, idx2song,
                    train, valid, fit, model, random_state):
    """
    Prepare data pools for training, validation and fit of the membership model.

    Parameters
    ----------
    playlists_idx: np array, shape (non-zeros, )
        Each element is a playlist index.
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    position: numpy array, shape (nnz, )
        Position of the corresponding song in the corresponding playlist.
    features: dict
        Song features indexed by MSD song id.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    train: numpy array, shape (split size, )
        Subsets the training playlists_idx, songs_idx and position coordinates.
    valid: numpy array, shape (split size, )
        Subsets the validation playlists_idx, songs_idx and position coords.
    valid: numpy array, shape (split size, )
        Subsets the fit playlists_idx, songs_idx and position coords.
    model: model file
        Used to further pass model-specific arguments.
    random_state: None, int or numpy RandomState
        Sets the pseudo-random behaviour.

    Returns
    -------
    dp_train: DataPool
        Data pool of training playlists.
    dp_valid: DataPool
        Data pool of validation playlists.
    dp_fit: DataPool
        Data pool of fit playlists.
    """

    # set random behavior
    rng = check_random_state(random_state)

    # prepare playlists as lists
    train_playlists = tolist(playlists_idx, songs_idx, position, idx2song, subset=train)
    valid_playlists = tolist(playlists_idx, songs_idx, position, idx2song, subset=valid)
    fit_playlists = tolist(playlists_idx, songs_idx, position, idx2song, subset=fit)

    # prepare playlists as data pools
    dp_train = DataPool(
        groups=train_playlists,
        mode=model.mode,
        features=features,
        standardize=model.standardize,
        normalize=model.normalize,
        random_state=rng
    )
    dp_valid = DataPool(
        groups=valid_playlists,
        mode=model.mode,
        features=features,
        standardize=model.standardize,
        scaler=dp_train.scaler,
        normalize=model.normalize,
        random_state=rng
    )
    dp_fit = DataPool(
        groups=fit_playlists,
        mode=model.mode,
        features=features,
        standardize=model.standardize,
        scaler=dp_train.scaler,
        normalize=model.normalize,
        random_state=rng
    )

    # enforce same out_size for train, valid and query data pools
    # on the cont data pool we evaluate songs, not playlists
    out_sizes = set([dp.out_size for dp in [dp_train, dp_valid, dp_fit]])

    if len(out_sizes) == 1:
        print('\nThe given arguments yield \'out_size\'={}.'.format(
            dp_train.out_size))

    else:
        out_size = min(out_sizes) if model.mode == 'fix' else max(out_sizes)
        print('\nThe given arguments don\'t yield playlists of equal length '
              'in both splits. Forcing \'out_size\' = {}'.format(out_size))

        # re-prepare data pools
        dp_train = DataPool(
            groups=train_playlists,
            mode=model.mode,
            out_size=out_size,
            features=features,
            standardize=model.standardize,
            normalize=model.normalize,
            random_state=rng
        )
        dp_valid = DataPool(
            groups=valid_playlists,
            mode=model.mode,
            out_size=out_size,
            features=features,
            standardize=model.standardize,
            scaler=dp_train.scaler,
            normalize=model.normalize,
            random_state=rng
        )
        dp_fit = DataPool(
            groups=fit_playlists,
            mode=model.mode,
            out_size=out_size,
            features=features,
            standardize=model.standardize,
            normalize=model.normalize,
            random_state=rng
        )

    return dp_train, dp_valid, dp_fit


if __name__ == '__main__':

    groups = [['a', 'b', 'c'], ['d', 'e']]
    features = {'a': np.array([0]), 'b': np.array([1]), 'c': np.array([2]),
                'd': np.array([3]), 'e': np.array([4]), 'f': np.array([5])}

    # pre-train a scaler
    X = np.array([[1, 1], [1, 1], [0, 0]])
    scaler = preprocessing.RobustScaler()
    scaler.fit(X)

    # data pool
    dp = DataPool(groups=groups, features=features, filter_size=0, mode='variable', out_size=1, random_state=1)

    print('min_size = {}'.format(dp.min_size))
    print('max_size = {}'.format(dp.max_size))
    print('out size = {}'.format(dp.out_size))
    print('original groups = {}'.format(dp.original_groups))
    print('groups = {}'.format(dp.groups))
    print('positive_pairs= {}'.format(dp.positive_pairs))
    U, m, u, v = dp[:]
    for i in range(dp.shape[0]):
        print('dp[{}] = {}, {}, {}, {}'.format(i, U[i], m[i], u[i], v[i]))


def select_model(model_path):
    """ Select model and related functions. """

    cfg_dir, model_dir = os.path.dirname(model_path).split('/')
    model_file = os.path.basename(model_path).split('.py')[0]

    model = False
    exec ('from {}.{} import {} as model'.format(cfg_dir, model_dir, model_file))
    model.name = model_file

    return model


def compute_membership_fix(playlists, idx2song, features, my_net, random_state):
    """
    Predict song-playlist probabilities given a membership model trained in
    mode='fix'. Loop over playlists, and for each playlist, compute its
    probabilities against all candidate songs at once.

    Parameters
    ----------
    playlists: list
        Each item is a playlist of variable length.
    idx2song: dict
        Mapping between song indices (used e.g., in arrays) and MSD song ids.
    features: dict
        Song features indexed by MSD song id.
    my_net: model
        Trained model in mode='fix'.
    random_state: None, int or numpy RandomState
        Sets the pseudo-random behaviour.

    Returns
    -------
    membership: numpy array, shape (num_songs, num_playlists)
        Song-playlist membership probabilities.
    """

    # set random behaviour
    rng = check_random_state(random_state)

    print('\nPredicting song-playlist probabilities...')

    # get dimensions

    # playlist length model takes as input (batch_size, pl_length, feat_size)
    input_size = my_net.input_shape[1]

    # number of candidate songs
    num_songs = len(idx2song)

    # stack batch with all the candidate songs features
    songs_feats = [features[idx2song[idx]].reshape(1, -1) for idx in range(num_songs)]
    songs_feats = np.vstack(songs_feats)

    # loop over playlists and evaluate against all candidate songs at once

    membership = []

    for pl_idx, pl in enumerate(tqdm(playlists)):

        # to deal with playlists of variable length using a model that takes
        # fix length input, sample fix-length sub-playlists of pl_length

        # number of possible sub-playlists from playlist
        num_subpl = misc.comb(len(pl), input_size, exact=True)

        # limit the number of sub-playlists if there are too many
        if num_subpl >= len(pl):
            subpl_iter = (rng.choice(a=pl, size=input_size, replace=False) for _ in range(len(pl)))
        else:
            subpl_iter = (list(draw) for draw in it.combinations(pl, input_size))

        # loop over fix-length sub-playlists

        output_list = []

        for subpl in subpl_iter:

            # shape sub-playlist features of shape (1, pl_length, feat_size)
            subpl_feats = [features[song].reshape(1, 1, -1) for song in subpl]
            subpl_feats = np.hstack(subpl_feats)

            # stack batch of repeated feats to evaluate against all songs
            subpl_feats = np.tile(subpl_feats, reps=(num_songs, 1, 1))

            # predict probabilities
            prob = my_net.predict_proba(
                [subpl_feats.astype(theano.config.floatX),
                 songs_feats.astype(theano.config.floatX)]
            )
            output_list.append(prob)

        # obtain playlist probs by aggregating the sub-playlists probs
        # hstack because prob has shape(num_songs, 1)
        membership.append(np.hstack(output_list).mean(axis=1, keepdims=True))

    # shape membership as songs-playlists array
    # hstack because each membership has shape(num_songs, 1)
    membership = np.hstack(membership)

    return membership
