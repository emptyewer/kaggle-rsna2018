# DeepRadiology Solution for RSNA Pneumonia Detection Challenge (Rank 7)

<span style="color:magenta">Install git lfs from [here](https://git-lfs.github.com/) to clone the repository</span>.

The solution files are organized in two directories `docker` and `cli`. This provides two ways for reproducing our final results for kaggle.

NOTE: Using `docker` is our recommended way to reproduce our competition results.

## Data Setup

1. Unzip the test images `stage_2_test_images.zip` to a convenient path. `<path>/stage_2_test_images` will be used in all the examples to refer to test images path.

2. Unzip the train images `stage_2_train_images.zip` to a convenient path. `<path>/stage_2_train_images` will be used in all the examples to refer to train images path.

3. Place models from `our_weights` folder in a convenient path. `<path>/weights` will be used in all the examples to refer to model/weights path. You can also leave the weights in the current path in the repository.

## System Setup and Requirements

NOTE: To clone the repository, make sure to have `git lfs` installed.

### Training
  - Ubuntu (16.04 recommended)
  - 4 GPUs (`NVIDIA GeForce GTX 1080Ti` or up)
  - Compatible nvidia drivers
  - `docker` (version 18.06.1+ recommended)
  - `nvidia-docker` v2.0+
  - `bash` v4.3+
  - `git lfs`

### Inference
  - Ubuntu (16.04 recommended)
  - At least 1 GPU (`NVIDIA GeForce GTX 1080Ti` or up)
  - Compatible nvidia drivers
  - `docker` (version 18.06.1+ recommended)
  - `nvidia-docker` v2.0+
  - `bash` v4.3+
  - `git lfs`
