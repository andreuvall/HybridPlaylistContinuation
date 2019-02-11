#!/usr/bin/env bash

# This script reproduces the results reported in Table 2, Figure 7a, Table 6
# and Table 12 of our paper
#   Vall et al. "Feature-Combination Hybrid Recommender Systems for Automated
#   Music Playlist Continuation." User Modeling and User-Adapted Interaction,
#   2019 (in press).
# It should also serve as a guide to get familiar with the code, reproduce the 
# rest of the results, and even conduct additional experiments.

# switch to the Python virtual environment
source venv/bin/activate

# create directory to output the results
mkdir -p results

# Profiles (run on gpu if possible)
python profiles_weak.py --model models/profiles/audio2cf_tags_logs.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python profiles_weak.py --model models/profiles/audio2cf_tags_logs.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/profiles_audio2cf_tags_logs
python profiles_weak.py --model models/profiles/logs.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python profiles_weak.py --model models/profiles/logs.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/profiles_logs
python profiles_weak.py --model models/profiles/audio2cf.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python profiles_weak.py --model models/profiles/audio2cf.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/profiles_audio2cf

# Membership (run on gpu if possible)
python membership_weak.py --model models/membership/audio2cf_tags_logs.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python membership_weak.py --model models/membership/audio2cf_tags_logs.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/membership_audio2cf_tags_logs
python membership_weak.py --model models/membership/logs.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python membership_weak.py --model models/membership/logs.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/membership_logs
python membership_weak.py --model models/membership/audio2cf.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python membership_weak.py --model models/membership/audio2cf.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/membership_audio2cf

# MF
python mf_weak.py --model models/mf/wmf.py --dataset data/aotm/ --msd data/MSD/ --fit --seed 1
python mf_weak.py --model models/mf/wmf.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --seed 1 > results/mf

# Hybrid MF
python mf_weak.py --model models/mf/wmf.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --precomputed logs --seed 1 > results/hmf_logs
python mf_weak.py --model models/mf/wmf.py --dataset data/aotm/ --msd data/MSD/ --test --song_occ 0 1 2 3 4 5 --precomputed audio2cf --seed 1 > results/hmf_audio2cf

# Neighbors
python neighbors_weak.py --user --dataset data/aotm/ --msd data/MSD/ --song_occ 0 1 2 3 4 5 > results/neighbors

# Artists
python neighbors_weak.py --artist --dataset data/aotm/ --msd data/MSD/ --song_occ 0 1 2 3 4 5 > results/artists

# CAGH
python neighbors_weak.py --artist --pop --dataset data/aotm/ --msd data/MSD/ --song_occ 0 1 2 3 4 5 > results/cagh

# Popularity
python popularity_weak.py --dataset data/aotm/ --msd data/MSD/ --song_occ 0 1 2 3 4 5 > results/popularity

# Random
python popularity_weak.py --dataset data/aotm/ --msd data/MSD/ --random --song_occ 0 1 2 3 4 5 > results/random

# turn off Python virtual environment
deactivate
