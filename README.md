# DISA: DIfferentiable Similarity Approximation for Universal Multimodal Registration

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_72) -- [ArXiv](https://arxiv.org/abs/2307.09931)

This repository contains the training code for our MICCAI 2023 paper [DISA: DIfferentiable Similarity Approximation for Universal Multimodal Registration](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_72)

## Data
Preprocessed training data extracted from the ["Gold Atlas - Male Pelvis - Gentle Radiotherapy" (Nyholm et al. 2017) dataset](https://doi.org/10.5281/zenodo.583096) can be downloaded from [Google Drive](https://drive.google.com/file/d/1AqhbgZHK-JL9qz_V_bJRWGhHGG44h0bU/view?usp=sharing).

The zip archive should be extracted in a folder called `Data` so that the npz files have a path `Data/*.npz`

## Training
```
docker build . -t disa
docker run -it --rm --gpus all --volume "$(pwd)/Data":/data --volume "$(pwd)/Output":/output --shm-size=32gb disa
```
After each epoch model weights are saved in the `Output` folder
