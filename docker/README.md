# Instructions for Docker

> Our final submission was generated after the `ensemble` step.

We provide a script `run.sh` to pull a prebuilt docker image and launch a properly configured jupyter notebook environment. We recommend using this script. You DO NOT have to build the docker image. We also provide the original `Dockerfile` in case you would like to build the image yourself.

NOTE: For training, we recommend 4 GPUs are used to reproduce our training results.

## Using `run.sh`

`run.sh` script has the following options.

```
Usage: bash run.sh --traindata=<path> --testdata=<path> [--gpus=GPU_IDs] [--share=SHARE_PATH]  [--weights=<path>]

   --traindata: path to the directory of train images

   --testdata: path to the directory of test images

   --gpus: GPU-IDs (separated by commas)
          default: 0,1,2,3

   --share: Additional paths to mount in docker.
            NOTE: supports multiple --share flags.

   --weights: path to the directory to save/load model file(s)

   --help: Show this help message.
```

> Running the script will launch jupyter notebook environment for doing Training, Prediction and Ensemble. As described below in section `Procedure`.

> `2222` port is exposed by docker which is mapped to `8888` for jupyter notebook server within docker.

> optionally one can pass GPU IDs by using the tag `--gpus`. By default, four GPUs `0,1,2,3` will be passed to the docker container. In order to simulate our training environment, we recommend you use 4 GPUs.

> `--weights` specifies where you want to save/load the model weights. For loading our trained weights, you should specify `<path>/weights`. By default, the model saves/loads weights in a subdir `weights` in the current working directory.

> `--share` tag is provided to add additional mount paths to do tests.

### Example

```
bash run.sh --traindata=<path>/stage_2_train_images --testdata=<path>/stage_2_test_images
```

## Procedure

Invoking `run.sh` script will launch jupyter notebook which exposes port `2222` on the host machine to port `8888` which is mapped to the jupyter notebook server inside docker.

Navigating to `http://<IP-ADDRESS>:2222` in a web-browser should list three jupyter notebooks.
NOTE: Ignore the folders listed in the notebooks root, they are necessary for proper execution of our code.

    1. Training.ipynb
    2. Predict.ipynb
    3. Ensemble.ipynb

### Step 1 (Training) - This step can be skipped if one would like to generate predictions directly from the models we provide

Simply run all the cells of `Training.ipynb` to train the model. This step will create four models after the training is complete, namely

```
couplenet_0_14_<batch>.pth
couplenet_1_14_<batch>.pth
couplenet_2_14_<batch>.pth
couplenet_3_14_<batch>.pth
```

NOTE: Make sure to `shutdown` the kernel of `Training.ipynb` before continuing. Otherwise `memory overflow` will occur.

### Step 2 (Predict)

Running all the cells of `Predict.ipynb` will generate prediction files from the saved models (either created from the above step or pre-loaded). These files will be saved to a `submissions` directory within the same directory as where `run.sh` command was invoked.

### Step 3 (Ensemble)

Running the cells of `Ensemble.ipynb` will generate a submission file by ensembling all four of the prediction files from the previous step. The file will be saved to an `ensemble` directory within the same directory as where `run.sh` command was invoked.

# Optional information for manual execution

## Docker Image

`venkykrishna/kaggle:rsna2018`

## Mount paths within docker container

* train_data (location of train dicoms): `/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTrain`
* test_data (location of test dicoms): `/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTest`
* model: `/notebooks/save/couplenet/res152/kaggle_pna`
* output: `/notebooks/output/couplenet/res152/kaggle_pna`

## internal port for jupyter notebook

`8888`

## Docker command

for Example:

`nvidia-docker run -v $HOME/train_data:/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTrain -v $HOME/test_data:/opt/R-FCN.pytorch/data/PNAdevkit/PNA2018/DCMImagesTest -v $HOME/model/:/notebooks/save/couplenet/res152/kaggle_pna -v $HOME/output/:/notebooks/output/couplenet/res152/kaggle_pna -p 2222:8888 venkykrishna/kaggle:rsna2018 /run_jupyter.sh`
