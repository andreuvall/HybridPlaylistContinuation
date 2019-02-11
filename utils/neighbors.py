# Neighbors utils.

from __future__ import print_function
from __future__ import division

from scipy import sparse

import numpy as np


def normalize_rowwise(m):
    """
    Divide every row of a binary sparse matrix by its norm.

    Parameters
    ----------
    m: scipy csr_matrix, shape (rows, cols)
        Binary matrix.

    Returns
    -------
    m_norm: numpy array, shape (rows, cols)
        Normalized version of `m`.
    """

    # compute per-row norm
    norm = np.sqrt(m.sum(axis=1))  # Y is binary: \sum 1^2 = \sum 1
    norm = sparse.csr_matrix(norm)  # save as sparse

    # build left-multiplying norm (check corresponding notebook)
    # summary: sparse arrays don't broadcast and something like
    # np.where(norm[:, na] == 0., 0., m / norm[:, na]) wouldn't work
    # we need to use the left-multiplying trick to achieve that
    data = 1. / norm.data
    indices = np.where(np.diff(norm.indptr) != 0)[0]
    indptr = norm.indptr
    rows = m.shape[0]
    left_norm = sparse.csr_matrix((data, indices, indptr), shape=(rows, rows))

    # compute row-wise normalized version of `m`
    m_norm = left_norm.dot(m)

    return m_norm


def compute_row_similarities(A):
    """
    Compute pairwise similarities between the rows of a binary sparse matrix.

    Parameters
    ----------
    A: scipy csr_matrix, shape (rows, cols)
        Binary matrix.

    Returns
    -------
    sim: numpy array, shape (rows, rows)
        Pairwise column similarities.
    """

    # normalize A in row-axis

    # 1) compute per-row norm
    norm = np.sqrt(A.sum(axis=1))  # Y is binary: \sum 1^2 = \sum 1
    norm = sparse.csr_matrix(norm)  # save as sparse

    # 2) build left-multiplying norm (https://stackoverflow.com/questions/16043299/substitute-for-numpy-broadcasting-using-scipy-sparse-csc-matrix)
    # summary: sparse arrays don't broadcast and something like
    # np.where(norm[:, na]==0., 0., A/norm[:, na]) wouldn't work
    # we need to use the left-multiplying trick to achieve that
    data = 1. / norm.data
    indices = np.where(np.diff(norm.indptr) != 0)[0]
    indptr = norm.indptr
    rows = A.shape[0]
    left_norm = sparse.csr_matrix((data, indices, indptr), shape=(rows, rows))

    # 3) compute row-wise normalized version of A
    A_norm = left_norm.dot(A)

    # compute pairwise row similarities
    sim = A_norm.dot(A_norm.T)

    return sim


def songcoo2artistcoo(songs_idx, idx2song, song2artist):
    """
    Express the playlists' song coordinates as artist coordinates.

    Parameters
    ----------
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    song2artist: dict
        Mapping between song ids in the MSD and artists.

    Returns
    -------
    sim: numpy array, shape (num songs, num songs)
        Artist-to-artist similarities moved to song-to-song matrix.
    """

    # build artist2idx dict for the playlist-artist matrices
    artists = sorted(list(set([song2artist[idx2song[sg_idx]] for sg_idx in songs_idx])))
    artist2idx = {a: i for i, a in enumerate(artists)}

    # 2) convert songs_idx coordinates to artists_idx coordinates
    artists_idx = np.array([artist2idx[song2artist[idx2song[idx]]] for idx in songs_idx])

    return artists, artist2idx, artists_idx


def artistsim2songsim(sim_aa, songs_idx, idx2song, song2artist, artist2idx):
    """
    Move artist-artist similarities to song-song similarities as follows: if
    sim(artist a, artist b) = s, then the sim(s_a, s_b) = s for all the
    songs s_a of artist a and all the songs s_b of artist b.


    Parameters
    ----------
    sim_aa: sparse csr_matrix, shape (num artists, num artists)
        Artist-to-artist similarities.
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    song2artist: dict
        Mapping between song ids in the MSD and artists.
    artist2idx: dict
        Mapping between artist and artist indices.

    Returns
    -------
    sim: sparse csr_matrix, shape (num songs, num songs)
        Artist-to-artist similarities moved to song-to-song matrix.
    """

    # revert artist2idx dict into idx2artist
    idx2artist = {i: a for a, i in artist2idx.iteritems()}

    # revert song2artist dict into artist2songs
    # (note that the result has many songs per artist)
    songs = set([idx2song[sg_idx] for sg_idx in songs_idx])
    artist2songs = {}
    for k, v in song2artist.iteritems():
        if k in songs:
            artist2songs[v] = artist2songs.get(v, [])
            artist2songs[v].append(k)

    # revert idx2song dict into song2idx
    song2idx = {s: i for i, s in idx2song.iteritems()}

    # traverse sim_aa and "re-distribute" the similarities

    row_list, col_list, data_list = [], [], []
    for r in xrange(sim_aa.shape[0]):
        # loop over all rows

        # find all songs from artist `r`
        songs_r = artist2songs[idx2artist[r]]
        idx_r = [song2idx[song] for song in songs_r]

        for index in xrange(sim_aa.indptr[r], sim_aa.indptr[r + 1]):
            # loop over all columns (in 'csr' style)

            # get col index `c` and similarity between artists `r` and `c`
            c = sim_aa.indices[index]
            sim_rc = sim_aa.data[index]

            # find all songs from artist `c`
            songs_c = artist2songs[idx2artist[c]]
            idx_c = [song2idx[song] for song in songs_c]

            # grow song-song similarity matrix in coo format
            # np.tile and np.repeat trick: given [1, 2, 3], [4, 5],
            # build [1, 2, 3, 1, 2, 3], [4, 4, 4, 5, 5, 5]
            rows = np.tile(idx_r, len(idx_c))
            cols = np.repeat(idx_c, len(idx_r))
            data = np.ones_like(rows) * sim_rc

            # accumulate
            row_list.append(rows)
            col_list.append(cols)
            data_list.append(data)

    # build song-song similarity matrix
    # (note that csr_matrix can be created with coordinates)
    row = np.hstack(row_list)
    col = np.hstack(col_list)
    data = np.hstack(data_list)
    num_songs = len(set(songs_idx))
    sim = sparse.csr_matrix((data, (row, col)), shape=(num_songs, num_songs))

    return sim


def artistscores2songscores(scores_a, songs_idx, idx2song, song2artist, artist2idx):
    """
    Move artist-artist similarities to song-song similarities as follows: if
    sim(artist a, artist b) = s, then the sim(s_a, s_b) = s for all the
    songs s_a of artist a and all the songs s_b of artist b.


    Parameters
    ----------
    scores_a: numpy array, shape (num artists, num playlists)
        Artist-playlist scores.
    songs_idx: np array, shape (non-zeros, )
        Each element is a song index.
    idx2song: dict
        Mapping between song indices and song ids in the MSD.
    song2artist: dict
        Mapping between song ids in the MSD and artists.
    artist2idx: dict
        Mapping between artist and artist indices.

    Returns
    -------
    scores_s: numpy array, shape (num songs, num playlists)
        Artist-playlist scores moved to song-playlist matrix.
    """

    # revert artist2idx dict into idx2artist
    idx2artist = {i: a for a, i in artist2idx.iteritems()}

    # revert song2artist dict into artist2songs
    # (note that the result has many songs per artist)
    songs = set([idx2song[sg_idx] for sg_idx in songs_idx])
    artist2songs = {}
    for k, v in song2artist.iteritems():
        if k in songs:
            artist2songs[v] = artist2songs.get(v, [])
            artist2songs[v].append(k)

    # revert idx2song dict into song2idx
    song2idx = {s: i for i, s in idx2song.iteritems()}

    # assign artists' scores to the corresponding songs
    scores_s = np.zeros((len(songs), scores_a.shape[1]))
    for r, scores in enumerate(scores_a):
        # loop over all rows

        # find all songs from artist `r`
        songs_r = artist2songs[idx2artist[r]]
        idx_r = [song2idx[song] for song in songs_r]

        # assign scores
        scores_s[idx_r] = scores

    return scores_s
