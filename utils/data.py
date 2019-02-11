# Data utils.

from __future__ import print_function
from __future__ import division

from scipy import sparse

import pandas as pd
import numpy as np

import cPickle
import os


def load_data(data_dir, msd_dir, model):
    """
    Load data.

    Parameters
    ----------
    data_dir: str
        Path to the playlists dataset directory.
    msd_dir: str
        Path to the MSD directory.
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
    split_weak: tuple
        Indices to obtain horizontal splits from pl_coo, namely:
            train_idx: numpy arrays, shape (train size, )
            valid_idx: numpy arrays, shape (valid size, )
            test_idx: numpy arrays, shape (test size, )
    split_strong: tuple
        Indices to obtain vertical splits from pl_coo, namely:
            train_idx: numpy arrays, shape (train size, )
            test_idx: numpy arrays, shape (test size, )
    features: dict
        Song features indexed by MSD song id.
    song2artist: dict
        Mapping between song ids in the MSD and artists.
    """

    print('\nLoading data...')

    # load playlists
    with open(os.path.join(data_dir, 'playlists.pkl'), 'r') as f:
        pl_coo = cPickle.load(f)

    # load weak split (horizontal)
    with open(os.path.join(data_dir, 'split_weak.pkl'), 'r') as f:
        split_weak = cPickle.load(f)

    # load strong split (vertical)
    with open(os.path.join(data_dir, 'split_strong.pkl'), 'r') as f:
        split_strong = cPickle.load(f)

    # load features
    if hasattr(model, 'feature'):
        feat_file = os.path.join(model.feature + '.pkl')
        with open(os.path.join(data_dir, 'features', feat_file), 'r') as f:
            features = cPickle.load(f)
    else:
        features = None

    # load song-artists just for information
    with open(os.path.join(msd_dir, 'song2artist.pkl'), 'r') as f:
        song2artist = cPickle.load(f)

    return pl_coo, split_weak, split_strong, features, song2artist


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
    print(('\t' + '\t{}' * 5).format('min', '1q', 'med', '3q', 'max'))
    print(('\t' + '\t{}' * 5).format(*hist_songs))

    hist_artists = pd.Series(art_playlists).describe()[['min', '25%', '50%', '75%', 'max']].tolist()
    print('\tPlaylists artists:')
    print(('\t' + '\t{}' * 5).format('min', '1q', 'med', '3q', 'max'))
    print(('\t' + '\t{}' * 5).format(*hist_artists))

    hist_items = pd.Series(song_occurrences).describe()[['min', '25%', '50%', '75%', 'max']].tolist()
    print('\tPlaylists per song:')
    print(('\t' + '\t{}' * 5).format('min', '1q', 'med', '3q', 'max'))
    print(('\t' + '\t{}' * 5).format(*hist_items))


def shape_data(playlists_idx, songs_idx, idx2song, features, mode='test',
               subset=None, verbose=True):
    """
    Prepare input array of song features and target array of song-playlists.

    Parameters
    ----------
    playlists_idx: np array, shape (non-zeros, )
        Each element is a playlist index.
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    features: dict
        Song features indexed by MSD song id.
    mode: str
        Either 'train' or 'test'. In 'train' mode we only use the songs that
        appear in the training split of the data. In 'test' mode we use all
        the songs in the dataset.
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

    if mode == 'train':
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


def show_data_splits(playlists_idx, songs_idx, idx2song, song2artist, train,
                     valid, fit, query, cont):
    """ Provide splits information. """

    print('\nTraining split:')
    compute_playlists_coo_stats(
        playlists_idx[train], songs_idx[train], idx2song, song2artist
    )
    print('\nValidation split:')
    compute_playlists_coo_stats(
        playlists_idx[valid], songs_idx[valid], idx2song, song2artist
    )
    print('\nFit split:')
    compute_playlists_coo_stats(
        playlists_idx[fit], songs_idx[fit], idx2song, song2artist
    )
    print('\nQuery split:')
    compute_playlists_coo_stats(
        playlists_idx[query], songs_idx[query], idx2song, song2artist
    )
    print('\nContinuation split:')
    compute_playlists_coo_stats(
        playlists_idx[cont], songs_idx[cont], idx2song, song2artist
    )
