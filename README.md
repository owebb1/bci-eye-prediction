# A Comparison of Least Angle Regression (LARS) to the Prediction Accuracy of Algorithms Tested on the EEGEyeNet Dataset

## Overview

This repository contains code and results from our investigation into the use of machine learning and deep learning models for predicting eye positions based on EEG data. We build upon the foundation set by EEGEyeNet by implementing and testing additional models—Least Angle Regressors (LARS) and Long Short-Term Memory networks (LSTM)—to explore their performance in this domain.

Our findings indicate that while LARS models fit well to high-dimensional EEG data, they do not outperform other machine learning models tested on EEGEyeNet, yielding an average RMS distance of 119.12 mm. Similarly, our LSTM models, though previously successful in motor imagery classification tasks, did not surpass the performance of CNNs for the prediction of absolute eye position in the context of EEG data.

For more details on our methodology and results, refer to our paper [least_angle_regression_eegeyenet.pdf](https://github.com/owebb1/bci-eye-prediction/blob/855487cd2a4b20bd4bfd77408a3bbcb710b1311f/least_angle_regression_eegeyenet.pdf).

This poster was accepted to HCII 2022. See the [acceptance email](https://github.com/owebb1/bci-eye-prediction/blob/855487cd2a4b20bd4bfd77408a3bbcb710b1311f/HCII_2022_Acceptance.pdf).

## Repository Contents

- `sequential/ProjectCode/src/*`: Minimal edits from the original sequential program code.
- `parallel/ProjectCode/src/*`: Contains all parallel adjustments, specifically in `network.cpp` and `neuralnet.cpp`.

## Prerequisites

Before running our code, ensure you have the following dependencies installed:

- python3
- numpy
- sklearn
- pyriemann
- scipy

## Environment Setup

To recreate our environment, follow the steps below. This setup includes PyTorch and other standard ML libraries as used for our final poster.

```bash
# Create and activate a new conda environment
conda create -n eegeyenet_LSTM python=3.8.5
conda activate eegeyenet_LSTM

# Install requirements
conda install --file general_requirements.txt
conda install --file standard_ml_requirements.txt

# Install PyTorch with or without CUDA
conda install pytorch torchvision torchaudio -c pytorch
# Use this line instead if CUDA is not installed
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Data Acquisition

You can download the necessary subset of the EEGEyeNet dataset from here. Navigate to the "prepared" folder and download the `Position_task_with_dots_synchronised_min_hilbert.npz` and `Position_task_with_dots_synchronised_min.npz` files. Place them in the `./data` directory within this repository.

## Recreate results

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

## Citation

Please cite our paper if you use our code or data for your research.

## Acknowledgments

We would like to extend our gratitude to the EEGEyeNet team for providing the comprehensive dataset that facilitated our research. Special thanks to the authors of the original EEGEyeNet Github repository, whose foundational work has been instrumental in our study.
