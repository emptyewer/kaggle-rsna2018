# Instruction for command line interface

> Our final submission was generated after the `ensemble` step.

## Prerequisites

> We highly recommend that you create a python virtual environment to run the following.

1. Create the virutalenv with python interpreter version `3.5.2` or up.

```
mkvirtualenv -p /usr/bin/python3
```

2. Install system libraries

```
apt install python-dev python3-dev libopencv-dev
```

3. add `CUDA` binaries to `PATH`

```
export PATH=$PATH:"/usr/local/cuda/bin"
```

4. Install `requirements.txt` for the package

```
pip install -r requirements.txt
```

5. Compile custom `cuda` code

```
cd lib
bash make.sh
cd ..
```

## Procedure

After setting up the environment, follow the below steps to generate our final submission file.

### Step 1 (Training)

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
The model files are named `couplenet_<checksession>_<checkepoch>_<checkpoint>.pth`.

### Predictions

```
python predict.py
```

This command reads the settings from `PREDICT_SETTINGS.json` to gather the locations of `TEST_DATA_CLEAN_PATH` images, `MODEL_DIR` input path and `SUBMISSION_DIR` output path.

Change `checksession`, `checkepoch`, `checkpoint` in `PREDICT_SETTINGS.json` to generate predictions for different models.

for e.g. to generate predictions from model `couplenet_10_14_6240.pth` modify

```
"checksession": 10,
"checkepoch": 14,
"checkpoint": 6420,
```

in the `PREDICT_SETTINGS.json` file.

NOTE: This script needs to be run multiple times to generate 4 different submissions for each of the 4 models that was generated from the previous step.

### Ensemble

```
python ensemble.py
```

This command reads from the file `ENSEMBLE_SETTINGS.json` to locate `SUBMISSION_DIR` input path and `ENSEMBLE_DIR` output paths. This script will read all the text files in the `SUBMISSION_DIR` and generate the ensemble.

NOTE: Keep all the settings same except for the directory paths in the `ENSEMBLE_SETTINGS.json` file.

*Note* Current implementation has a bug, different order of the prediction files will produce slightly different final results. The file order to produce our exact final result is: `couplenet_4_14_6670.pth`, `couplenet_3_14_6670.pth`, `couplenet_1_14_6670.pth`, `couplenet_2_14_6670.pth`. We are working to correct this bug and will update the repository accordingly.
