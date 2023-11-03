# A Comparison of Least Angle Regression (LARS) to the Prediction Accuracy of Algorithms Tested on the EEGEyeNet Dataset


## Overview

This code is based off of the [EEGEyeNet Github](https://github.com/ardkastrati/EEGEyeNet) which is also linked on their webpage [EEGEyeNet](http://eegeye.net). In the README_EEGEyeNet are the instructions included by them to run the code, but this README will describe what is needed to recreate our results. To add additional models, refer to the README_EEGEyeNet. 

## Getting Started

### Prerequisites

The follwing are necessary for our paper, but can easily be installed using the enviorment installation below. 
- python3
- numpy
- sklearn
- pyriemann
- scipy

### Environment Installation
The following is based off of the environment setup described in [EEGEyeNet](http://eegeye.net)

For our Final poster, we require an environment that contains pytorch and other standard ML models.

General Requirements
1. Create a new conda environment:

```bash
conda create -n eegeyenet_LSTM python=3.8.5 
```

2. Activate the environment first
```bash
conda activate eegeyenet_LSTM
```
3. Install the general_requirements.txt

```bash
conda install --file general_requirements.txt 
```

4. Install the Standard ML Requirements

```bash
conda install --file standard_ml_requirements.txt 
```

5. Install pytorch requirments

    - If cuda is installed (suggested), run :

    ```bash
    conda install pytorch torchvision torchaudio -c pytorch
    ```
    - If cuda not installed, run:
     ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```


### Download Data
The data can be located on [EEGEyeNet](http://eegeye.net) under "data", or directly at [EEGEyeNet Data](https://osf.io/ktv7m/).

Our file only uses a subsection of the data, so there is no reason to download the entirety of the dataset.

The steps to download the data are included below:
1. Visit [EEGEyeNet Data](https://osf.io/ktv7m/)
2. Go into the "prepared" folder
3. Click on `Position_task_with_dots_synchronised_min_hilbert.npz` which brings you to a new page. In the top right, hit download.
4. Follow step 3 for the file `Position_task_with_dots_synchronised_min.npz`.
5. Place these files in a folder titled `./data`

### Recreate results

Recreating our results assuming the environment is set up, the data is placed in the `./data` directory and the config is left untouched, is simple.

To reproduce all of the results in Table 1 including EEGEyeNet machine learning models (roughly 2.5 hours):

1. From the `code` directory, run `python3 main.py`.
2. The location of the results will be printed to the terminal but can be found in `code/runs/`. The directory that is formed in `./runs` for your unique run will contain `runs.csv`. This will contain each of the models, runtimes, and rms accuracy in pixels.
3. The results conatained in `runs.csv` are in pixels, so with two pixels per millimeter, divide the pixel result by 2 to recreate the results used in the poster.

To reproduce the results in Table 2 includeing EEGEyeNet deep learning models:

1. From the `code` directory, open `./config.py`
2. Set `config["feature_extraction"] = False`
3. Set `config["include_ML_models"] = False`
4. Set `config["include_DL_models"] = True`
5. Set `config["include_your_ml_models"] = False`
6. Set `config["include_your_DL_models"] = True`
7. Set `config["include_dummy_models"] = False`
8. Save `./config.py`
9. run `python3 main.py`

To reproduce only our LARS model results (Roughly 30 seconds): 
1. From the `code` directory, open `./config.py`
2. Set `config["feature_extraction"] = True`
3. Set `config["include_ML_models"] = False`
4. Set `config["include_DL_models"] = False`
5. Set `config["include_your_ml_models"] = True`
6. Set `config["include_your_DL_models"] = False`
7. Set `config["include_dummy_models"] = False`
8. Save `./config.py`
9. run `python3 main.py`

To reporoduce only our LSTM model results:
1. From the `code` directory, open `./config.py`
2. Set `config["feature_extraction"] = False`
3. Set `config["include_ML_models"] = False`
4. Set `config["include_DL_models"] = False`
5. Set `config["include_your_ml_models"] = False`
6. Set `config["include_your_DL_models"] = True`
7. Set `config["include_dummy_models"] = False`
8. Save `./config.py`
9. run `python3 main.py`



