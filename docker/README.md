# Instructions for Docker

Make sure to setup the system and download the proper files  mentioned in the `README.md` file in the parent directory.

We provide a convience script `run.sh` to pull a prebuilt docker image and launch the notebook environment that contains all the required code and configurations.

NOTE:

> Our final submission was generated after the `ensemble`

You DO NOT have to build the docker image. We also provide the original `Dockerfile` in case you would like to build the image yourself.

## Convinience script `run.sh`

The script has the following options.

Port exposed by docker `2222` which is mapped to `8888` within docker.

```
Usage: bash run.sh --traindata=<path> --testdata=<path> [--gpus=GPU_IDs] [--share=SHARE_PATH]  [--weights=<path>]

   --share: Additional paths to mount in docker.
            NOTE: supports multiple --share flags.
   --gpus: GPU-IDs (separated by commas)
          default: 0,1

   --traindata: path to the directory of train images

   --testdata: path to the directory of test images

   --weights: path to the directory of model file(s)

   --help: Show this help message.
```

Running the script will launch a jupyter notebook environment for doing Training, Prediction and Ensemble. As described below in section `Procedure`.

## Example

```
bash run.sh --traindata=<path>/stage_2_train_images --testdata=<path>/stage_2_test_images
```

(optionally) one can provide the GPU IDs using the tag `--gpus`. By default, two GPUS `0,1` will be passed to the docker container.

NOTE: Setting `--weights` option is only necessary if you would like to skip the traning steps and generate submissions directly from our models. `--weights` should point to the path where the weights (`.pth` files) from `our_models` are located.

NOTE: `--share` is provided for convinience if you would like to do some tests.

## Procedure

### Step 1 (Training)

Invoking `run.sh` script will launch jupyter notebook which exposes port `2222` on the host machine to port `8888` which is mapped to the jupyter notebook server inside docker

navigating to `http://<IP-ADDRESS>:2222` in a web-browser should list three jupyter notebooks

    1. Training.ipynb
    2. Predict.ipynb
    3. Ensemble.ipynb

Simply run all the cells of `Training.ipynb` to train the model. The generated models will be output to a folder called `weights` in the same location where `run.sh` was invoked.

NOTE: This step can be skipped if one would like to generate predictions directly from the models we provide

### Step 2.

NOTE: Make sure to `shutdown` the kernel of `Training.ipynb` before continuing. Otherwise `memory overflow` will occur.

Running all the cells of `Predict.ipynb` will recreate the submission file in the leaderboard. This file will be output to directory named `submissions` within the current working directory.

### Step 3.

Running the cells of `Ensemble.ipynb` will recreate the submission file by ensembling a number of predictions. This file will be output to directory named `ensemble` within the current working directory.

# Optional information for manual execution

## Docker Image

`venkykrishna/kaggle:rsna2018`

## Mount paths within docker container

* data: `/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTest`
* model: `/notebooks/save/couplenet/res152/kaggle_pna`
* output: `/notebooks/output/couplenet/res152/kaggle_pna`

## internal port for jupyter notebook

`8888`

## Docker command

for Example:

`nvidia-docker run -v $HOME/data:/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTest -v $HOME/model/:/notebooks/save/couplenet/res152/kaggle_pna -v $HOME/output/:/notebooks/output/couplenet/res152/kaggle_pna -p 2222:8888 venkykrishna/kaggle:rsna2018 /run_jupyter.sh`
