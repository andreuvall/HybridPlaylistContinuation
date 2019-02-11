from __future__ import print_function
from __future__ import division

from utils.data import load_data, show_data_splits, shape_data
from utils.evaluation import mask_array_rows, evaluate
from utils.mf import select_model, show_design, load_mf, train, fit

import implicit

import time
import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

'''
Collaborative Filtering baseline for music playlist continuation based on the
factorization of the playlist-song matrix of co-occurrences. We use the
Weighted Matrix Factorization algorithm. Once the factorization is learned, it
is used to populate a matrix of playlist-song scores describing how well a
playlist and a song fit together. Thus, a playlist can be extended by selecting
the songs with highest score.

In this program we explore the so-called weak generalization setting. That is,
the factorization is trained on the same playlists that will be extended.

This approach is purely collaborative in the usual sense in the recommender
systems literature, i.e., it only relies on the collaborative information given
by the actual playlists examples. However, it is also possible to partly skip
the factorization of the playlist-song matrix and use independently
pre-computed song factors, either derived from the factorization of independent
listening logs or predicted by a pre-trained audio2cf network similar to
van den Oord et al. "Deep Content-Based Music Recommendation". (This latter
option is then a hybrid variant, in the sense that it combines collaborative
and audio information.)

If pre-computed song factors are used, we have seen that playlist factors
derived from the factorization of the playlist-song matrix don't work too good.
Instead, we obtain appropriate playlist factors by performing "half" an
iteration of Alternating Least Squares given the playlist-song matrix and the
independently pre-computed song factors.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Collaborative music playlist continuation based on matrix factorization.')
    parser.add_argument('--model', type=str, help='path to the model specification file', metavar='')
    parser.add_argument('--dataset', type=str, help='path to the playlists dataset directory', metavar='')
    parser.add_argument('--msd', type=str, help='path to the MSD directory', metavar='')
    parser.add_argument('--train', action='store_true', help='train the matrix factorization with monitoring')
    parser.add_argument('--fit', action='store_true', help='fit the matrix factorization')
    parser.add_argument('--test', action='store_true', help='evaluate the playlist continuations')
    parser.add_argument('--precomputed', type=str, default=None, help='name of pre-computed song factors to test, if not None', metavar='')
    parser.add_argument('--song_occ', type=int, help='test on songs observed song_occ times during training', nargs='+', metavar='')
    parser.add_argument('--ci', action='store_true', help='compute confidence intervals if True')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU if True')
    parser.add_argument('--metrics_file', type=str, help='file name to save metrics', metavar='')
    parser.add_argument('--seed', type=int, help='set random behavior', metavar='')
    args = parser.parse_args()

    # set random behavior
    # (only randomness is in `implicit`, which only depends on numpy)
    if args.seed is not None:
        np.random.seed(args.seed)

    # set model configuration
    model = select_model(args.model)

    # prepare output directory
    data_name = os.path.basename(os.path.normpath(args.dataset))
    out_dir = os.path.join('params', 'mf', model.name + '_' + data_name + '_weak')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data: playlists, splits and artist info
    if args.precomputed is not None:
        model.feature = args.precomputed
    data = load_data(args.dataset, args.msd, model)
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

    # provide model information
    print('\nMatrix Factorization:')
    show_design(model)

    if args.train:
        #
        # train the matrix factorization while validating on withheld playlists
        #

        # prepare song-playlist matrix in training
        _, Y_train = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=train_idx
        )

        # prepare song-playlist matrix in validation
        _, Y_valid = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=valid_idx
        )

        # train the model
        train(
            model=model,
            train_target=Y_train,
            valid_target=Y_valid,
            out_dir=out_dir,
            use_gpu=args.use_gpu
        )

    if args.fit:
        #
        # fit the matrix factorization model
        #

        # prepare song-playlist matrix in full training
        _, Y_fit = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=fit_idx
        )

        # fit the model
        fit(
            model=model,
            fit_target=Y_fit,
            out_dir=out_dir,
            use_gpu=args.use_gpu
        )

    if args.test:
        #
        # extend the playlists in the query split and evaluate the
        # continuations by comparing them to actual withheld continuations
        #

        # load previously learned latent factors
        params_path = os.path.join(out_dir, model.name + '_params.pkl')
        songs, playlists = load_mf(params_path)

        # prepare song-playlist matrix in test continuations
        if args.precomputed is None:
            # use `songs` factors obtained by song-playlist factorization
            _, Y_cont = shape_data(
                playlists_idx, songs_idx, idx2song=None, features=None,
                subset=cont_idx
            )
        else:
            # overwrite `songs` with pre-computed factors from logs or audio2cf
            songs, Y_cont = shape_data(
                playlists_idx, songs_idx, idx2song=idx2song, features=features,
                subset=cont_idx
            )

        # prepare song-playlist matrix in test queries
        # used to mask songs from queries and to calculate playlist factors
        _, Y_query = shape_data(
            playlists_idx, songs_idx, idx2song=None, features=None,
            subset=query_idx
        )

        # calculate number of song occurrences when MF was fit
        # used to discard unknown songs and for cold-start analysis
        # fit_idx = query_idx --> Y_fit = Y_query
        train_occ = np.asarray(Y_query.sum(axis=1)).flatten()

        start = time.time()

        if args.precomputed is not None:
            # `if args.retrain_playlists and (args.precomputed is not None)`
            # would use previously computed playlists factors. I've tried it
            # and the results are very bad. This makes sense, because the
            # solution of the factorization is "very non-unique", and the
            # playlist and the song factors need to be "well-coupled" to work

            # compute playlist factors adapted to pre-computed song factors with
            # half iter. of ALS given the song factors and the "query" playlists
            mf = implicit.als.AlternatingLeastSquares(
                factors=model.num_factors,
                regularization=model.l2_weight,
                use_cg=False,
                use_gpu=args.use_gpu,
                iterations=1
            )
            mf.item_factors = songs.astype('float32')
            mf.fit(item_users=model.positive_weight * Y_query, show_progress=False)
            playlists = mf.user_factors

        # predict song-playlist scores
        print('\nPredicting song-playlist scores...')
        # start = time.time()
        cont_output = songs.dot(playlists.T)
        print('\nTime predicting: {} sec.'.format(round(time.time() - start, 4)))

        if args.precomputed is None:
            # mask song-playlist continuation pairs involving unknown songs
            mask_array_rows(Y_cont, np.where(train_occ == 0)[0])

        # evaluate the continuations
        evaluate(
            scores=[cont_output.T],
            targets=[Y_cont.T.tocsr()],
            queries=[Y_query.T.tocsr()],
            train_occ=[train_occ],
            k_list=[10, 30, 100],
            ci=args.ci,
            song_occ=args.song_occ,
            metrics_file=args.metrics_file
        )
