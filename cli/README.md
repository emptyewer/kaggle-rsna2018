# Instruction for command line interface

> Our final submission was generated after the `ensemble`

## Prerequisites

1. We highly recommend that you create a python virtual environment to run this code.

2. Create the virutalenv with python interpreter version `3.5.2` or up.

```
mkvirtualenv -p /usr/bin/python3
```

3. Install system libraries

```
apt install python-dev python3-dev libopencv-dev
```

4. add `CUDA` binaries to `PATH`

```
export PATH=$PATH:"/usr/local/cuda/bin"
```

5. Install `requirements.txt` for the package

```
pip install -r requirements.txt
```

6. Compile custom `cuda` code

```
cd lib
bash make.sh
cd ..
```

## Procedure

After setting up the environment, run the follow the three below steps to generate our final submission

### Training

```
python train.py
```

This command reads the file `TRAIN_SETTINGS.json` to gather the locations of `TRAIN_DATA_CLEAN_PATH` images and `MODEL_DIR` output path. Do not change other params in the `json` file.

This will generate 4 models, namely

```
couplenet_0_14_<batch>.pth
couplenet_1_14_<batch>.pth
couplenet_2_14_<batch>.pth
couplenet_3_14_<batch>.pth
```

### Predictions

```
python predict.py
```

This command reads from the file `PREDICT_SETTINGS.json` to read the locations of `TEST_DATA_CLEAN_PATH` images, `MODEL_DIR` input path and `SUBMISSION_DIR` output path.

The model files are named `couplenet_<checksession>_<checkepoch>_<checkpoint>.pth`. Change these vars  in `PREDICT_SETTINGS.json` to generate predictions for different models.

for e.g. to generate predictions from model `couplenet_10_14_6240.pth` modify

```
"checksession": 10,
"checkepoch": 14,
"checkpoint": 6420,
```

in the `PREDICT_SETTINGS.json` file.

NOTE: This file has to be run multiple times to generate 4 different submissions for each model that was generated above.

### Ensemble

```
python ensemble.py
```

This command reads from the file `ENSEMBLE_SETTINGS.json` to locate `SUBMISSION_DIR` input path and `ENSEMBLE_DIR` output paths. This script will read all the text files in the `SUBMISSION_DIR` and output the ensembles.

NOTE: Keep all the settings same except for the directory paths in the `ENSEMBLE_SETTINGS.json` file.
