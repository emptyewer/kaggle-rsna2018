# Instructions for Docker

> Our final submission was generated after the `ensemble` step.

Make sure to follow the instructions for `Data setup` and `System setup` from the `README.md` in the parent directory.

We provide a script `run.sh` to pull a prebuilt docker image and launch a properly configured jupyter notebook environment. We recommend using this script.

NOTE:

You DO NOT have to build the docker image. We also provide the original `Dockerfile` in case you would like to build the image yourself.

## Using `run.sh`

`run.sh` script has the following options.

```
Usage: bash run.sh --traindata=<path> --testdata=<path> [--gpus=GPU_IDs] [--share=SHARE_PATH]  [--weights=<path>]

   --traindata: path to the directory of train images

   --testdata: path to the directory of test images

   --gpus: GPU-IDs (separated by commas)
          default: 0,1

   --share: Additional paths to mount in docker.ok
            NOTE: supports multiple --share flags.

   --weights: path to the directory of model file(s)

   --help: Show this help message.
```

> Running the script will launch jupyter notebook environment for doing Training, Prediction and Ensemble. As described below in section `Procedure`.

> `2222` port is exposed by docker which is mapped to `8888` for jupyter notebook server within docker.

> optionally one can pass GPU IDs by using the tag `--gpus`. By default, two GPUS `0,1` will be passed to the docker container.

> `--weights` option is only necessary to skip the traning steps and generate submissions directly from our models. This option should point to the path `<path>/weights` where the weights (`.pth` files) downloaded from `our_weights` are located.

> `--share` tag is provided to add additional mount paths to do tests.

### Example

```
bash run.sh --traindata=<path>/stage_2_train_images --testdata=<path>/stage_2_test_images
```

## Procedure

Invoking `run.sh` script will launch jupyter notebook which exposes port `2222` on the host machine to port `8888` which is mapped to the jupyter notebook server inside docker.

Navigating to `http://<IP-ADDRESS>:2222` in a web-browser should list three jupyter notebooks. NOTE: Ignore the folders listed in the notebooks root, they are necessary for proper execution of our code.

    1. Training.ipynb
    2. Predict.ipynb
    3. Ensemble.ipynb

### Step 1 (Training)

Simply run all the cells of `Training.ipynb` to train the model. The generated models will be output to a folder called `weights` in the same location where `run.sh` was invoked. This step will output four models after the training is complete, namely

```
couplenet_0_14_<batch>.pth
couplenet_1_14_<batch>.pth
couplenet_2_14_<batch>.pth
couplenet_3_14_<batch>.pth
```

NOTE: This step can be skipped if one would like to generate predictions directly from the models we provide

NOTE: Make sure to `shutdown` the kernel of `Training.ipynb` before continuing. Otherwise `memory overflow` will occur.

### Step 2 (Predict)

Running all the cells of `Predict.ipynb` will generate prediction files for each of the 4 models from above. These files will be output to a directory named `submissions` within the current working directory (from where `run.sh` script was invoked).

### Step 3 (Ensemble)

Running the cells of `Ensemble.ipynb` will recreate the submission file by ensembling all four of the prediction files from the previous step. This file will placed into a directory named `ensemble` within the current working directory (from where `run.sh` script was invoked).

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
