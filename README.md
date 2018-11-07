# ChallengePneumo Solution for RSNA Pneumonia Detection Challenge (Rank 7)

The solution files are organized in two directories `docker` and `cli`. This provides two ways for reproducing our final results for kaggle.

NOTE: Using `docker` is our recommended way to reproduce our competition results.

## Data Setup

1. Unzip the test images `stage_2_test_images.zip` to a convenient path. `<path>/stage_2_test_images` will be used in all the examples to refer to test images path.

2. Unzip the train images `stage_2_train_images.zip` to a convenient path. `<path>/stage_2_train_images` will be used in all the examples to refer to train images path.

3. Download models from `our_weights` folder in this repository and place it a convenient path. `<path>/weights` will be used in all the examples to refer to model/weights path.

## System Setup and Requirements

### Training
  - Ubuntu (16.04 reccomended) 
  - At least 4 GPUs (`NVIDIA GeForce GTX 180Ti` or up)
  - Compatable nvidia drivers
  - `docker` (version 18.06.1+ reccomended)
  - `nvidia-docker` v2.0+
  - `bash` v4.3+
  - `git lfs`

### Inference
  - Ubuntu (16.04 reccomended) 
  - At least 1 GPU (`NVIDIA GeForce GTX 180Ti` or up)
  - Compatable nvidia drivers
  - `docker` (version 18.06.1+ reccomended)
  - `nvidia-docker` v2.0+
  - `bash` v4.3+
  - `git lfs`

## Team Members

* Adil Ainihaer
* Ankoor Bhagat
* Venkatramanan Krishnamani
* Tong Luo
* Jameson Merkow
* Arvind M Vepa
