# Data utils.

from __future__ import print_function
from __future__ import division

from scipy import sparse
from collections import Counter

import theano
import pandas as pd
import numpy as np

import cPickle
import os


def load_data(data_dir, model):
    """
    Load data.

    Parameters
    ----------
    data_dir: str
        Path to the data directory.
    model: model file
        Indicates the song features we need to load.

    Returns
    -------
    pl_coo: tuple
        Coordinates to build the playlist-song sparse matrix, namely:
        playlists_idx: np array, shape (non-zeros, )
            Each element is a playlist index.
        songs_idx: np array, shape (non-zeros, )
            Each element is a song index.
        idx2song: dict
            Mapping between song indices and song ids in the MSD.
    split_idx: tuple
        Indices to obtain data splits from pl_coo, namely:
            train_idx: numpy arrays, shape (train size, )
            valid_idx: numpy arrays, shape (valid size, )
            test_idx: numpy arrays, shape (test size, )
    features: dict
        Mapping between song ids and their feature representation.
    song2artist: dict
        Mapping between song ids in the MSD and artists.
    """

    print('\nLoading data...')

    # load playlists
    with open(os.path.join(data_dir, 'playlists.pkl'), 'r') as f:
        pl_coo = cPickle.load(f)

    # load train/valid/test splits
    with open(os.path.join(data_dir, 'splits.pkl'), 'r') as f:
        split_idx = cPickle.load(f)

    # load features
    if model.mode is 'hybrid':
        feat_file = os.path.join(model.feature + '.pkl')
        with open(os.path.join(data_dir, 'song_features', feat_file), 'r') as f:
            features = cPickle.load(f)
    else:
        features = None

    # load song-artists just for information
    with open(os.path.join('data', 'MSD', 'song2artist.pkl'), 'r') as f:
        song2artist = cPickle.load(f)

    return pl_coo, split_idx, features, song2artist


def compute_playlists_coo_stats(playlists_idx, songs_idx, idx2song, song2artist):
    """
    Compute basic stats for playlists encoded in coordinate format.

    Parameters
    ----------
    playlists_idx: np array, shape (non-zeros, )
        Each element is a playlist index.
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    song2artist: dict
        Mapping between song ids in the MSD and artists.
    """

    playlists = pd.DataFrame(
        {'playlist': playlists_idx,
         'song': songs_idx,
         'artist': [song2artist[idx2song[idx]] for idx in songs_idx]}
    )

    # compute stats
    len_playlists = playlists['playlist'].value_counts().values
    count_playlists = len(len_playlists)
    song_occurrences = playlists['song'].value_counts().values
    count_songs = len(song_occurrences)
    count_artists = len(playlists['artist'].unique())
    art_playlists = playlists.groupby(['playlist'])['artist'].nunique().values

    # show stats
    print('\tFound {} playlists, {} artists, {} songs.'.format(
        count_playlists, count_artists, count_songs))

    hist_songs = pd.Series(len_playlists).describe()[['min', '25%', '50%', '75%', 'max']].tolist()
    print('\tPlaylists length:')
    print(('\t\t{:<2}' * 5).format('min', '1q', 'med', '3q', 'max'))
    print(('\t\t{:<2.3}' * 5).format(*hist_songs))

    hist_artists = pd.Series(art_playlists).describe()[['min', '25%', '50%', '75%', 'max']].tolist()
    print('\tPlaylists artists:')
    print(('\t\t{:<2}' * 5).format('min', '1q', 'med', '3q', 'max'))
    print(('\t\t{:<2.3}' * 5).format(*hist_artists))

    hist_items = pd.Series(song_occurrences).describe()[['min', '25%', '50%', '75%', 'max']].tolist()
    print('\tPlaylists per song:')
    print(('\t\t{:<2}' * 5).format('min', '1q', 'med', '3q', 'max'))
    print(('\t\t{:<2.3}' * 5).format(*hist_items))


def shape_data(playlists_idx, songs_idx, idx2song, features, mode, subset=None,
               verbose=False):
    """
    This function prepares the data for the hybrid classifier, namely
    the input array of song features and the target array of song-playlists.
    In 'fit' mode we only use the songs that appear in the training split of
    the data. In 'test' mode we use all the songs in the dataset.

    Parameters
    ----------
    playlists_idx: np array, shape (non-zeros, )
        Each element is a playlist index.
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    features: dict
        Mapping between song ids and their feature representation.
    mode: str
        Either 'fit' or 'test'.
    subset: None or numpy array, shape (split size, )
        Subsets the playlists_idx and songs_idx coordinates.
    verbose: bool
        Print info if True.

    Returns
    -------
    X: numpy array, shape (num_songs, feature size)
        Input feature representation for each song.
    Y: sparse csr_matrix, shape (num_songs, num playlists)
        Target playlists each song belongs to.
    """

    if verbose:
        print('\nShaping data...')

    if subset is None:
        subset = range(len(playlists_idx))

    # get full playlist-song dimensions
    num_playlists = len(np.unique(playlists_idx))
    num_songs = len(np.unique(songs_idx))

    if mode == 'fit':
        # use only the songs in the subset
        unique_songs = np.unique(songs_idx[subset])
    else:
        # use all songs
        unique_songs = np.unique(songs_idx)

    # input: features of each song
    if features is not None:
        X = np.vstack([features[idx2song[idx]] for idx in unique_songs])
    else:
        # dummy empty features for matrix cf, to keep the interface
        X = np.empty(1)

    # target: binary array of playlists each song belongs to
    Y = sparse.csr_matrix(
        (np.ones_like(subset), (songs_idx[subset], playlists_idx[subset])),
        shape=(num_songs, num_playlists)
    )
    Y = Y[unique_songs]

    return X, Y
