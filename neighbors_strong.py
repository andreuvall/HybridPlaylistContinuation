from __future__ import print_function
from __future__ import division

from scipy import sparse

from utils.data import load_data, show_data_splits, shape_data
from utils.evaluation import mask_array_rows, evaluate
from utils.neighbors import normalize_rowwise, songcoo2artistcoo, artistsim2songsim

import argparse
import os

import numpy as np


'''
Collaborative filtering baseline for music playlist continuation based on
playlist neighborhoods.

In this program we explore the so-called strong generalization setting. That is,
the playlist neighbors are computed on playlists different than those extended.
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

    # load data: playlists, splits and artist info
    data_name = os.path.basename(os.path.normpath(args.dataset))
    data = load_data(args.dataset, args.msd, None)
    playlists_coo, split_weak, split_strong, features, song2artist = data

    # playlists_coo are the playlists stored in coordinate format
    playlists_idx, songs_idx, _, idx2song = playlists_coo

    # split_weak provides a query/continuation split
    train_idx_cnt, test_idx_cnt = np.hstack(split_weak[:2]), split_weak[2]

    cont_output_l, Y_cont_l, Y_query_l, train_occ_l = [], [], [], []
    for fold in range(5):

        print('\nRunning fold {}...'.format(fold))

        # split_strong defines a playlist-disjoint split
        # chose one of the folds
        fold_strong = split_strong[fold]
        train_idx_dsj, test_idx_dsj = np.hstack(fold_strong[:2]), fold_strong[2]

        # define splits for this experiment
        # train model on the intersection of disjoint training and queries
        # validate model on the inters. of disjoint training and continuations
        # fit the model on the disjoint training playlists
        # extend only the playlist-disjoint test split
        train_idx = np.intersect1d(train_idx_dsj, train_idx_cnt)
        valid_idx = np.intersect1d(train_idx_dsj, test_idx_cnt)
        fit_idx = train_idx_dsj
        query_idx = np.intersect1d(test_idx_dsj, train_idx_cnt)
        cont_idx = np.intersect1d(test_idx_dsj, test_idx_cnt)

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

        # prepare song-playlist matrix in fit playlists
        _, Y_fit = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=fit_idx
        )

        # calc. song occs. in "fit" playlists, where neighbors were calculated
        # used to discard unknown songs and for cold-start analysis
        train_occ = np.asarray(Y_fit.sum(axis=1)).flatten()

        # compute co-occurrence similarities

        if args.user:
            # user-based collaborative filtering
            # i.e., `sim` contains playlist-to-playlist similarities

            if args.artist:
                # playlist similarities based on artist co-occurrences

                # convert songs_idx coordinates to artists_idx coordinates
                artists, artist2idx, artists_idx = songcoo2artistcoo(songs_idx, idx2song, song2artist)

                # prepare artist-playlist matrix in query playlists
                _, Y_artist_query = shape_data(
                    playlists_idx, artists_idx, idx2song=None, features=None,
                    subset=query_idx
                )

                # prepare artist-playlist matrix in fit playlists
                _, Y_artist_fit = shape_data(
                    playlists_idx, artists_idx, idx2song=None, features=None,
                    subset=fit_idx
                )

                # normalize query playlists and fit playlists
                # (using Y_query.T the similarities are computed column-wise,
                # but the output needs to be transposed back)
                Y_artist_query_norm = normalize_rowwise(Y_artist_query.T).T
                Y_artist_fit_norm = normalize_rowwise(Y_artist_fit.T).T

                # compute the cosine between fit and query playlists
                # resulting array has non-empty shape (num_fit x num_query),
                # even though the sparse array has full shape num_pl x num_pl
                sim = Y_artist_fit_norm.T.dot(Y_artist_query_norm)

            else:
                # playlist similarities based on song co-occurrences

                # normalize query playlists and fit playlists
                # (using Y_query.T the similarities are computed column-wise,
                # but the output needs to be transposed back)
                Y_query_norm = normalize_rowwise(Y_query.T).T
                Y_fit_norm = normalize_rowwise(Y_fit.T).T

                # compute the cosine between fit and query playlists
                # resulting array has non-empty shape (num_fit x num_query),
                # even though the sparse array has full shape num_pl x num_pl
                sim = Y_fit_norm.T.dot(Y_query_norm)

        else:
            # item-based collaborative filtering
            # i.e., `sim` contains song-to-song similarities

            if args.artist:
                # song similarities based on artist co-occurrences

                # convert songs_idx coordinates to artists_idx coordinates
                artists, artist2idx, artists_idx = songcoo2artistcoo(songs_idx, idx2song, song2artist)

                # prepare artist-playlist matrix in fit playlists
                _, Y_artist_fit = shape_data(
                    playlists_idx, artists_idx, idx2song=None, features=None,
                    subset=fit_idx
                )

                # normalize fit playlists
                Y_artist_fit_norm = normalize_rowwise(Y_artist_fit)

                # compute song-to-song similarities in fit playlists
                sim_aa = Y_artist_fit_norm.dot(Y_artist_fit_norm.T)

                # move artist-artist similarities to song-song similarities
                sim = artistsim2songsim(sim_aa, songs_idx, idx2song, song2artist, artist2idx)

            else:
                # song similarities based on song co-occurrences

                # normalize fit playlists
                Y_fit_norm = normalize_rowwise(Y_fit)

                # compute "fit"-song to "query"-song similarities
                sim = Y_fit_norm.dot(Y_fit_norm.T)

        # predict song-playlist scores
        if args.user:
            # why Y_fit?
            # Y_fit contains the 'fit' playlists where songs have occurred
            # sim contains query-to-fit playlist similarities
            # -> the product assigns high score to songs that have occurred in
            # fit playlists similar to the query playlists
            cont_output = Y_fit.dot(sim).toarray()
        else:
            # why Y_query?
            # sim contains song-to-song similarities derived from fit playlists
            # Y_query contains the query playlists where songs have occurred
            # -> the product assigns high score to songs similar to songs that
            # have occurred in the query playlists
            cont_output = sim.dot(Y_query).toarray()

        # factor by popularity if required
        if args.pop:
            cont_output *= train_occ[:, np.newaxis]

        # mask song-playlist continuation pairs involving unknown songs
        mask_array_rows(Y_cont, np.where(train_occ == 0)[0])

        # append arrays re-shaping for evaluation
        cont_output_l.append(cont_output.T)
        Y_cont_l.append(Y_cont.T.tocsr())
        Y_query_l.append(sparse.csr_matrix(Y_query).T.tocsr())
        train_occ_l.append(train_occ)

    # evaluate the continuations
    evaluate(
        scores=cont_output_l,
        targets=Y_cont_l,
        queries=Y_query_l,
        train_occ=train_occ_l,
        k_list=[10, 30, 100],
        ci=args.ci,
        song_occ=args.song_occ,
        metrics_file=args.metrics_file
    )
