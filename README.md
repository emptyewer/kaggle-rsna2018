# ChallengePneumo Solution for RSNA Pneumonia Detection Challenge (Rank 7)

The solution files are organized in two directories `docker` and `cli`. This provides two ways for reproducing our final results for kaggle.

NOTE: The `docker` way is the most reliable way of reproducing our results. The setup and config are a bit involved for the `cli`.

## Data Setup

1. Unzip the test images `stage_2_test_images.zip` to a convinient path. `<path>/stage_2_test_images` will be used for test images path.

2. Unzip the train images `stage_2_train_images.zip` to a convinient path. `<path>/stage_2_train_images` will be used for train images path.

3. Download models from `our_models` folder in this repository and place it a convinient path. `<path>/weights` will be used in all the examples to refer to model/weights path.

## System Setup and Requirements

1. Atleast one GPU (`NVIDIA GeForce GTX 180Ti` or up) and 4 CPUs.

2. install proper `nvidia` drivers for GPUs.

3. install and configure `docker` and `nvidia-docker2`.

4. `bash` or `zsh` shell
