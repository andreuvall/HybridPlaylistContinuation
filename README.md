Music Playlist Continuation by Learning from Hand-Curated Examples and Song Features
====================================================================================

This repository contains the implementation of the hybrid playlist continuation model presented in the following submission to the ECML-PKDD 2017 conference:

* Andreu Vall, Hamid Eghbal-zadeh, Matthias Dorfer and Markus Schedl. “[Music Playlist Continuation by Learning from Hand-Curated Examples and Song Features.](http://www.cp.jku.at/research/papers/Vall_etal_ISMIR_2015.pdf)”

The submission presents results on the "[AotM-2011](https://bmcfee.github.io/data/aotm2011.html)"  and the "[8tracks](https://8tracks.com/)" datasets, both enriched with song features derived from the [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/) (MSD).

The "AotM-2011" dataset and the MSD are publicly available. Therefore, we can share the exact playlists and song features used in our experiments ([download the AotM-2011 dataset](http://www.cp.jku.at/datasets/recommendation/data_HybridPlaylistContinuation.zip)). Confer the submission for full details on the data preparation. Also note that we share the already derived features. The contribution of this work does not reside on the feature extraction step, but on the enhanced effect of combining hand-curated music playlists with song features.

 The 8tracks dataset is a private collection given to us for research purposes and we are not allowed to disclose it.

## Models

The repository contains two main programs: `playlist_hybrid.py` and `playlist_cf.py`. The former implements our hybrid playlist continuation model. The latter is the Collaborative Filtering (CF) baseline used in the submission, based on the weighted matrix factorization algorithm introduced in [Hu et al. 2008](http://yifanhu.net/PUB/cf.pdf) and implemented in the [`implicit`](https://github.com/benfred/implicit) package.

## Basic Usage

The proposed model can be trained using, e.g., song features derived from listening logs:
```bash
python playlist_hybrid.py --config config/hybrid/logs.py --fit
```

Once the model is trained the playlist continuations it produces can be evaluated by running:
```bash
python playlist_hybrid.py --config config/hybrid/logs.py --test
```

## Song-to-playlist Classifier

The proposed hybrid playlist continuation model is powered by a song-to-playlist classifier that, on the basis of song features and hand-curated playlist examples, predicts if a song is a good continuation for a given playlist. The classifier is implemented in `utils/song2play.py`.

Its CF counterpart is implemented in `utils/mf.py` and relies on a [slight modification of the `implicit`](https://github.com/andreuvall/implicit) package.


## Configuration Files

The model architecture, the hyperparameters and the set of song features used in different instances of the proposed hybrid model are specified through configuration files. The depth of the factorization and the hyperparameters used in the CF baseline are also specified this way.

The configuration files must reside in the `config` directory, organized as follows:
```
config
 +-- cf
      +-- wmf.py
 +-- hybrid
      +-- ivectors_songtags.py
      +-- ivectors.py
      ...
```
The configuration for the CF baseline (`playlist_cf.py`) must be located in the `config/cf` subdirectory. The configuration for the proposed hybrid model (`playlist_hybrid.py`) must be located in the `config/hybrid` subdirectory.

To use a configuration file provide its path to the `--config` option of the main programs `playlist_hybrid.py` and `playlist_cf.py`.

The default configuration files provided in the repository should reproduce the submission results.

## Data

[Download the AotM-2011 dataset](http://www.cp.jku.at/datasets/recommendation/data_HybridPlaylistContinuation.zip). Decompress it and place the obtained `data` directory at the root level of the repository.

## License

The contents of this repository are licensed. See the LICENSE file for further details.
