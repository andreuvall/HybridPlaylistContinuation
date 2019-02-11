from __future__ import print_function
from __future__ import division

from scipy import sparse

from utils.data import load_data, show_data_splits, shape_data
from utils.evaluation import mask_array_rows, evaluate
from utils.neighbors import compute_row_similarities, songcoo2artistcoo, artistsim2songsim

import argparse
import os
import time

import numpy as np

'''
Collaborative filtering baseline for music playlist continuation based on
playlist neighborhoods with similarities based on artist co-occurrences.

In this program we explore the so-called weak generalization setting. That is,
the playlist neighbors are computed on the same playlists that will be extended.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Neighborhoods-based collaborative filtering for music playlist continuation.')
    parser.add_argument('--user', action='store_true', help='user-based if true, else item-based')
    parser.add_argument('--artist', action='store_true', help='artist-based similarities if True, else song-based')
    parser.add_argument('--pop', action='store_true', help='factor by popularity if True')
    parser.add_argument('--dataset', type=str, help='path to the playlists dataset directory', metavar='')
    parser.add_argument('--msd', type=str, help='path to the MSD directory', metavar='')
    parser.add_argument('--ci', action='store_true', help='compute confidence intervals if True')
    parser.add_argument('--song_occ', type=int, help='test on songs observed song_occ times during training', nargs='+', metavar='')
    parser.add_argument('--metrics_file', type=str, help='file name to save metrics', metavar='')
    args = parser.parse_args()

    # settings
    cf_mode = 'user' if args.user else 'item'
    sim_mode = 'artist' if args.artist else 'song'

    # load data: playlists, splits and artist info
    data_name = os.path.basename(os.path.normpath(args.dataset))
    data = load_data(args.dataset, args.msd, None)
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

    #
    # extend the playlists in the query split and evaluate the
    # continuations by comparing them to actual withheld continuations
    #

    # prepare song-playlist matrix in test continuations
    _, Y_cont = shape_data(
        playlists_idx, songs_idx, idx2song=None, features=None,
        subset=cont_idx
    )

    # prepare song-playlist matrix in test queries
    # used to mask songs from queries
    _, Y_query = shape_data(
        playlists_idx, songs_idx, idx2song=None, features=None,
        subset=query_idx
    )

    # calculate song occs. in "query" playlists, where neighbors were calculated
    # used to discard unknown songs and for cold-start analysis
    # (in weak setting query and fit playlists are the same)
    train_occ = np.asarray(Y_query.sum(axis=1)).flatten()

    # compute co-occurrence similarities
    start = time.time()

    if args.user:
        # user-based collaborative filtering
        # i.e., `sim` contains playlist-to-playlist similarities

        if args.artist:
            # playlist similarities based on artist co-occurrences

            # convert songs_idx coordinates to artists_idx coordinates
            artists, artist2idx, artists_idx = songcoo2artistcoo(songs_idx, idx2song, song2artist)

            # prepare artist-playlist matrix in query playlists
            # used to compute artist-artist similarities
            _, Y_artist = shape_data(
                playlists_idx, artists_idx, idx2song=None, features=None,
                subset=query_idx
            )

            # compute artist-artist similarities
            sim = compute_row_similarities(Y_artist.T)

        else:
            # playlist similarities based on song co-occurrences
            # (using Y_query.T the similarities are computed column-wise)
            sim = compute_row_similarities(Y_query.T)

    else:
        # item-based collaborative filtering
        # i.e., `sim` contains song-to-song similarities

        if args.artist:
            # song similarities based on artist co-occurrences

            # convert songs_idx coordinates to artists_idx coordinates
            artists, artist2idx, artists_idx = songcoo2artistcoo(songs_idx, idx2song, song2artist)

            # prepare artist-playlist matrix in query playlists
            # used to compute artist-artist similarities
            _, Y_artist = shape_data(
                playlists_idx, artists_idx, idx2song=None, features=None,
                subset=query_idx
            )

            # compute artist-artist similarities
            sim_aa = compute_row_similarities(Y_artist)

            # move artist-artist similarities to song-song similarities
            sim = artistsim2songsim(sim_aa, songs_idx, idx2song, song2artist, artist2idx)

        else:
            # song similarities based on song co-occurrences
            sim = compute_row_similarities(Y_query)

    print('\nComputed \"{}\"-based similarities from {} co-occurrences '
          '({:.4f} sec.).'.format(cf_mode, sim_mode, time.time() - start))

    # predict song-playlist scores
    start = time.time()
    if args.user:
        cont_output = Y_query.dot(sim).toarray()
    else:
        cont_output = sim.dot(Y_query).toarray()

    # factor by popularity if required
    if args.pop:
        cont_output *= train_occ[:, np.newaxis]

    print('\nPredicted song-playlist scores ({:.4f} sec.).'.format(time.time() - start))

    # mask song-playlist continuation pairs involving unknown songs
    mask_array_rows(Y_cont, np.where(train_occ == 0)[0])

    # evaluate the continuations
    evaluate(
        scores=[cont_output.T],
        targets=[Y_cont.T.tocsr()],
        queries=[sparse.csr_matrix(Y_query).T.tocsr()],
        train_occ=[train_occ],
        k_list=[10, 30, 100],
        ci=args.ci,
        song_occ=args.song_occ,
        metrics_file=args.metrics_file
    )
