Hybrid recommender systems for music playlist continuation
==========================================================

This repository contains Python implementations of the hybrid recommender systems Profiles and Membership introduced in our paper

- Andreu Vall, Matthias Dorfer, Hamid Eghbal-zadeh, Markus Schedl, Keki Burjorjee, and Gerhard Widmer. "Feature-Combination Hybrid Recommender Systems for Automated Music Playlist Continuation." User Modeling and User-Adapted Interaction, 2019 (in press).

_(If you arrived to this repository from another of our papers, please look at the end of this README file.)_

The repository also implements the baselines MF, Hybrid MF, Neighbors, Artists, CAGH, Popularity and Random (we use the terminology of the paper).

According to the evaluation methodology followed in the paper, the recommender systems are implemented in "weak" and "strong" generalization. For example, MF is implemented by `mf_weak.py` and `mf_strong.py`. The only exception is Profiles, which, as is, only operates in weak mode.

## Basic usage

The proposed hybrid systems (e.g., Profiles) can be trained using any type of song feature vectors (e.g., Logs features):
```bash
python profiles_weak.py --model models/profiles/logs.py --dataset data/aotm/ --msd data/MSD/ --fit
```

Once the system is trained, the playlist continuations it produces can be evaluated by running:
```bash
python profiles_weak.py --model models/profiles/logs.py --dataset data/aotm/ --msd data/MSD/ --test
```

While Profiles and Membership have a similar interface, the baseline recommender systems may have different options. Details about any system implementation (e.g., about Popularity) can be obtained by running:
```bash
python popularity_strong.py --h
```

Importantly, note that:

- Hybrid MF is implemented by `mf_weak.py` and `mf_strong.py` but using the `precomputed` option to indicate if "Audio2CF" or "Logs" features should be used instead of the song factors derived from the factorization of the playlist-song matrix.
- Artists is implemented by `neighbors_weak.py` and `neighbors_strong.py` but passing the flag `artist`, which switches from song-level to artist-level similarities.
- CAGH is implemented by `neighbors_weak.py` and `neighbors_strong.py` but passing the flags `artist` and `pop`, where the former switches from song-level to artist-level similarities, and the latter weights the final playlist-song scores by the song popularity.
- Random can be evaluated by passing the flag `random` to `popularity_weak.py` and `popularity_strong.py` (the flag simply overrides the usual behavior of the scripts).

## Model files

Profiles, Membership, MF and Hybrid MF require model specifications regarding the number of unknowns in the systems, the song features considered, whether regularization should be used, etc. These are specified by model specification files. We suggest to place the model specification files in the `models` directory, organized as follows:
```
models
+-- profiles
    +-- audio2cf.py
    +-- songtags.py
    +-- cf.py
    ...
+-- membership
    +-- audio2cf.py
    +-- songtags.py
    +-- cf.py
    ...
+-- mf
    +-- wmf.py
...
```

The model configuration files provided in the repository should reproduce the results reported in the paper.

## Set up

The required Python packages are listed in the `requirements.txt` file. I recommend running the dedicated script 
```bash
source setup_env.sh
```
to create a Python virtual environment and take care of the requirements. It is important to note that Profiles and Membership are implemented using [Lasagne](https://lasagne.readthedocs.io/en/latest/#) and [Theano](http://deeplearning.net/software/theano/). These libraries will likely stop evolving (see [Theano's announcement](https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ)) and may require specific (older) versions of packages like [NumPy](http://www.numpy.org) or [SciPy](https://www.scipy.org/).

It may also be necessary to install [pygpu](http://deeplearning.net/software/libgpuarray) for GPU support. Running the following script should manually install the package and its dependencies in the newly created virtual environment:
```bash
source setup_pygpu.sh
```

## Data

The paper presents a thorough off-line evaluation conducted on two playlist datasets: the publicly available [AotM-2011](https://bmcfee.github.io/data/aotm2011.html) dataset (derived from the [Art of the Mix](http://www.artofthemix.org) platform), and a private collection that [8tracks](https://8tracks.com/) shared with us for research purposes. The playlist collections are enriched with song features derived from the publicly available [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/).

We share the filtered playlists and song features corresponding to the AotM-2011 collection. We can not share any information related to the 8tracks collection. [Download the data](http://drive.jku.at/ssf/s/readFile/share/8197/5021896040269493362/publicLink/data_HybridPlaylistContinuation_.zip), decompress it, and place the obtained `data` directory at the root level of the repository.

## Reproducing the results 

Table 2, Figure 7a, Table 6, and part of Table 12 of the paper can be reproduced by running the dedicated script:
```bash 
source reproduce_results.sh
```
For simplicity, the script only reports the central performance values and not the confidence intervals (which can be obtained passing the flag `ci`). 

The script outputs a file for each recommender system. Ideally one should become familiar with the code to properly interpret these results. Importantly, for each system, several similar-looking tables will be reported. This is because at test time we pass `0 1 2 3 4 5` to the `song_occ` option, which takes care of the following:

- we first obtain the overall results (as in Table 2), then
- we obtain the results on songs that occurred 0, 1, 2, 3, 4 or 5+ times at training time (as in Figure 7a), then
- we obtain the results for songs that occurred 4- times at training time (as in Table 12), and finally
- we obtain the results for songs that occurred in 1+ times at training time (that is, in-set songs, as in Table 6).

This script does not reproduce all the results reported in the paper (e.g., strong generalization is missing) but it should provide enough examples to get familiar with the code, to reproduce the remaining experiments, and even to conduct new experiments.

## License

The contents of this repository are licensed. See the LICENSE file for further details.

## Previous versions

You may have arrived to this repository following the link from our previous paper:

- Andreu Vall, Hamid Eghbal-zadeh, Matthias Dorfer, Markus Schedl, and Gerhard Widmer. "Music Playlist Continuation by Learning from Hand-Curated Examples and Song Features: Alleviating the Cold-Start Problem for Rare and out-of-Set Songs." In Proc. DLRS Workshop at RecSys, 46-54. Como, Italy, 2017.

The newer version of the repository encompasses the previous, and the data shared now is almost identical to that used in the previous paper (only the training/test splits in weak generalization have changed). 

You can also browse the previous version of the repository (tagged as `DLRS2017`) by [clicking here](https://github.com/andreuvall/HybridPlaylistContinuation/tree/a7612175de82faae003c3b309472e85700818d05), or you can check it out by running:
```bash
 git checkout DLRS2017
``` 

If you do check it out, beware of the behavior of `git` [checking out tags](https://git-scm.com/book/en/v2/Git-Basics-Tagging). 
