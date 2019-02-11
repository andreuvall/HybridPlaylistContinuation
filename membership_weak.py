from __future__ import print_function
from __future__ import division

from sklearn.utils import check_random_state
from lasagne_wrapper.network import Network

from utils.evaluation import evaluate
from utils.data import load_data, shape_data, show_data_splits
from utils.membership import tolist, shape_datapools, select_model, compute_membership_fix

import numpy as np

import argparse
import os
import time

'''
Hybrid music playlist continuation based on playlist-song membership.
We learn a binary classifier to decide if any playlist-song pair (represented 
by feature vectors) is a good match or not. Once it is learned, such classifier 
is used to populate a matrix of song-playlist scores describing how well a song
and a playlist fit together. Thus, a playlist can be extended by selecting
the songs with highest score. 

This approach is "hybrid" in the usual sense in the recommender systems 
literature, i.e., it combines content (given by the song features) and 
collaborative information (given by playlists examples).

In this program we explore the so-called weak generalization setting. That is, 
the membership model is trained on the same playlists that will be extended.
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hybrid music playlist continuation based on playlist-song membership.')
    parser.add_argument('--model', type=str, help='path to the model specification file', metavar='')
    parser.add_argument('--dataset', type=str, help='path to the playlists dataset directory', metavar='')
    parser.add_argument('--msd', type=str, help='path to the MSD directory', metavar='')
    parser.add_argument('--fit', action='store_true', help='fit the song-to-playist classifier with monitoring')
    parser.add_argument('--test', action='store_true', help='evaluate the playlist continuations')
    parser.add_argument('--ci', action='store_true', help='compute confidence intervals if True')
    parser.add_argument('--song_occ', type=int, help='test on songs observed song_occ times during training', nargs='+', metavar='')
    parser.add_argument('--metrics_file', type=str, help='file name to save metrics', metavar='')
    parser.add_argument('--seed', type=int, help='set random behavior', metavar='')
    args = parser.parse_args()

    # set random behavior
    rng = check_random_state(args.seed)

    # load model configuration
    model = select_model(args.model)

    # prepare output directory
    data_name = os.path.basename(os.path.normpath(args.dataset))
    out_dir = os.path.join('params', 'membership', model.name + '_' + data_name + '_weak')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data: playlists, splits, features and artist info
    data = load_data(args.dataset, args.msd, model)
    playlists_coo, split_weak, split_strong, features, song2artist = data

    # playlists_coo are the playlists stored in coordinate format
    playlists_idx,  songs_idx, position, idx2song = playlists_coo

    # split_strong defines a playlist-disjoint split
    # this is just to validate the model, use any disjoint split
    fold_strong = split_strong[0]
    train_idx_dsj, test_idx_dsj = np.hstack(fold_strong[:2]), fold_strong[2]

    # split_weak provides a query/continuation split
    query_idx, cont_idx = np.hstack(split_weak[:2]), split_weak[2]

    # define splits for this experiment
    # train model on intersection of disjoint training split and queries
    # validate model on intersection of disjoint test split and queries
    # fit the model on all the queries
    # extend all the playlists, using all queries and continuations
    train_idx = np.intersect1d(train_idx_dsj, query_idx)
    valid_idx = np.intersect1d(test_idx_dsj, query_idx)
    fit_idx = query_idx
    # query_idx = query_idx
    # cont_idx = cont_idx

    # shape data pools for the membership model
    dp_train, dp_valid, dp_fit = shape_datapools(
        playlists_idx=playlists_idx,
        songs_idx=songs_idx,
        position=position,
        features=features,
        idx2song=idx2song,
        train=train_idx,
        valid=valid_idx,
        fit=fit_idx,
        model=model,
        random_state=rng
    )

    # provide data information
    show_data_splits(playlists_idx, songs_idx, idx2song, song2artist,
                     train_idx, valid_idx, fit_idx, query_idx, cont_idx)

    if args.fit:
        #
        # fit the membership model while validating on withheld playlists
        #

        # define network
        print('\nCreating model ...')
        net = model.create_model(
            batch_size=model.BATCH_SIZE,
            pl_length=dp_train.out_size - 1,
            feat_dim=model.FEAT_DIM,
        )

        # initialize network
        my_net = Network(net)

        # compile training data
        data = {'train': dp_train, 'valid': dp_valid, 'test': dp_valid}

        # set up output files
        params_file = os.path.join(out_dir, '{}_params.pkl'.format(model.name))
        log_file = os.path.join(out_dir, '{}_log_train.pkl'.format(model.name))

        # train model
        start = time.time()
        my_net.fit(
            data=data,
            training_strategy=model.train_strategy,
            dump_file=params_file,
            log_file=log_file
        )
        print('\nTime fitting: {:.4f} sec.'.format(time.time() - start))

    if args.test:
        #
        # extend the playlists in the query split and evaluate the
        # continuations by comparing them to actual withheld continuations
        #

        # define network (batch_size = num_songs, to evaluate all candidates)
        print('\nCreating model ...')
        net = model.create_model(
            batch_size=len(idx2song),
            pl_length=dp_train.out_size - 1,
            feat_dim=model.FEAT_DIM,
        )

        # initialize network
        my_net = Network(net)

        # load previously fit parameters
        params_file = os.path.join(out_dir, '{}_params.pkl'.format(model.name))
        my_net.load(file_path=params_file)

        # shape withheld continuations for evaluation
        _, Y_cont = shape_data(
            playlists_idx, songs_idx, idx2song, features, subset=cont_idx
        )

        # songs in the "query" playlists need to be masked to make sure that
        # they are not recommended as continuations
        _, Y_query = shape_data(
            playlists_idx, songs_idx, idx2song, features, subset=query_idx
        )

        # get num of song occurrences when model was fit for cold-start analysis
        # Y_fit = Y_query
        train_occ = np.asarray(Y_query.sum(axis=1)).flatten()

        # convert query playlists to list
        query_playlists = tolist(playlists_idx, songs_idx, position, idx2song, subset=query_idx)

        # predict song-playlist probabilities
        start = time.time()
        cont_output = compute_membership_fix(
            playlists=query_playlists,
            idx2song=idx2song,
            features=features,
            my_net=my_net,
            random_state=rng
        )
        print('\nTime predicting: {} sec.'.format(round(time.time() - start, 4)))
        # cont_output = np.random.rand(len(idx2song), len(query_playlists))

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
