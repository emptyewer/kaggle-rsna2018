# ChallengePneumo Solution for RSNA Pneumonia Detection Challenge (Rank 7)

<span style="color:magenta">Install git lfs from [here](https://git-lfs.github.com/) to clone the repository</span>.

The solution files are organized in two directories `docker` and `cli`. This provides two ways for reproducing our final results for kaggle.

NOTE: Using `docker` is our recommended way to reproduce our competition results.

## Data Setup

1. Unzip the test images `stage_2_test_images.zip` to a convenient path. `<path>/stage_2_test_images` will be used in all the examples to refer to test images path.

2. Unzip the train images `stage_2_train_images.zip` to a convenient path. `<path>/stage_2_train_images` will be used in all the examples to refer to train images path.

3. Download models from `our_weights` folder in this repository and place it a convenient path. `<path>/weights` will be used in all the examples to refer to model/weights path.

## System Setup and Requirements

1. Atleast one GPU (`NVIDIA GeForce GTX 180Ti` or up). 4 GPUs recommended.

2. Install proper `nvidia` drivers for GPUs.

3. Install and configure `docker` and `nvidia-docker2`.

4. `git lfs` can be downloaded from [here](https://github.com/git-lfs/git-lfs/releases/tag/v2.6.0).

4. `bash` or `zsh` shell

## Team Members

* Adil Ainihaer
* Ankoor Bhagat
* Venkatramanan Krishnamani
* Tong Luo
* Jameson Merkow
* Arvind M Vepa
