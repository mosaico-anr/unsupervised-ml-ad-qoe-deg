# Which ML Model for Detecting QoE Degradation in Low-Latency Applications: Cloud-Gaming Case Study

This repository contains the code of the different unsupervised machine learning algorithms implemented in the paper to detect anomalies in Cloud Gaming Sessions.

The models implemented are :
- [Isolation Forest](https://ieeexplore.ieee.org/document/4781136)
- [One Class SVM](http://nips.djvuzone.org/djvu/nips12/0582.djvu)
- [Deep-SVDD](http://proceedings.mlr.press/v80/ruff18a.html)
- [PCA]()
- [Auto-Encoders]()
- [LSTM-VAE](https://doi.org/10.1109/LRA.2018.2801475)
- [DAGMM](https://openreview.net/forum?id=BJJLHbb0-)
- [USAD](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392)

## Disclaimer

The paper is under review at the Special Issue on Robust and Reliable Networks of the Future in IEEE Transactions on Network and Service Management. The repository is for reviewers only and will be publicly available upon paper acceptance.

## Datasets

The datasets used in this paper are from 03 Cloud Gaming platforms: 
  - [Google Stadia](https://stadia.google.com/gg/) (shut down since January 2023)
  - [NVIDIA GeForceNow](https://www.nvidia.com/fr-fr/geforce-now/)
  - [Microsoft XCloud](https://www.xbox.com/fr-FR/cloud-gaming)
  
  They can be download on [this link](https://cloud-gaming-traces.lhs.loria.fr/ANR-19-CE25-0012_std_gfn_xc_cg_webrtc_metrics.7z).  
Download and unzip the file in the data folder.

```bash
mkdir data
cd data/
tar -xvf ANR-19-CE25-0012_std_gfn_xc_cg_webrtc_metrics.tar.xz .
```

## Dependencies

The main dependencies are the following:

- Python 3.8+
- Torch
- Tensorflow
- Numpy
- Matplotlib
- Pandas
- Scikit-Learn

## Installation

We first recommand you to create a Python env to install the required packages.

```bash
python -m venv [ENV_NAME] python=3.8
pip install -r requirements.txt
```

## Usage

Install Python modules to avoid absolute import errors.

```bash
python setup.py install --user
```

From the root of the repository, the usage of the main file can be seen by running:

```bash
python -m main --help
```

This will output the following parameters to run the main program that allow you to test each of the eight models used in the paper.

```bash
usage: main.py [-h] --model-name {PCA,OC-SVM,IForest,AE,LSTM-VAE,DAGMM,Deep-SVDD,USAD} [--window-size WINDOW_SIZE] [--contamination-ratio CONTAMINATION_RATIO]
               [--seed SEED] [--model-save-path MODEL_SAVE_PATH] [--data-dir DATA_DIR] [--platform {std,gfn,xc}] [--threshold THRESHOLD]
               [--wad-threshold WAD_THRESHOLD] --metric {pw,window_wad,window_pa,window_pak,window_rpa} [--is-trained]

optional arguments:
  -h, --help            show this help message and exit
  --model-name {PCA,OC-SVM,IForest,AE,LSTM-VAE,DAGMM,Deep-SVDD,USAD}
                        The model to train and test.
  --window-size WINDOW_SIZE
                        The window size. Default is 10.
  --contamination-ratio CONTAMINATION_RATIO
                        The contamination ratio. Default is 0.
  --seed SEED           The random generator seed. Default is 42.
  --model-save-path MODEL_SAVE_PATH
                        The folder to store the model outputs.
  --data-dir DATA_DIR   The folder where the data are stored.
  --platform {std,gfn,xc}
                        The platform data to use. Default is std.
  --threshold THRESHOLD
                        The threshold of anomalous observations to consider in a window. Default is 0.8.
  --wad-threshold WAD_THRESHOLD
                        WAD approach (alpha) threshold. Default is 0.8.
  --metric {pw,window_wad,window_pa,window_pak,window_rpa}
                        The metric to use for evaluation.
  --is-trained          If the models are already trained. Default action is false.

```

The hyper-parameters of each model used in the paper are the default parameters in the code. They can be seen or changed in their respective code file (`src/models/[model_name].py`).

## Papers experiments

To reproduce the experiments of the paper, refer to `src/experiments/` folder.
You can for instance, reproduce the experiments of data contamination impact using `src/experiments/data_contamination.py` file as follows:
```bash
python -m src.experiments.data_contamination 
```

The aforementioned file take the following arguments from command line.

```bash
usage: data_contamination.py [-h] [--window-size WINDOW_SIZE] [--threshold THRESHOLD] [--wad-threshold WAD_THRESHOLD] [--contamination-ratio CONTAMINATION_RATIO]
                             [--platform {std,gfn,xc}] [--n-runs N_RUNS] [--models-list [{PCA,IForest,OC-SVM,AE,Deep-SVDD,USAD,DAGMM,LSTM-VAE} ...]]
                             [--save-dir SAVE_DIR] [--data-dir DATA_DIR] [--is-trained]

optional arguments:
  -h, --help            show this help message and exit
  --window-size WINDOW_SIZE
                        The window size. Default is 10.
  --threshold THRESHOLD
                        The threshold of anomalous observations to consider in a window. Default is 0.8.
  --wad-threshold WAD_THRESHOLD
                        WAD approach (alpha) threshold. Default is 0.8.
  --contamination-ratio CONTAMINATION_RATIO
                        The contamination ratio. Default is 0.
  --platform {std,gfn,xc}
                        The platform data to use. Default is std.
  --n-runs N_RUNS       The number of time each experiment is runned. Default is 5.
  --models-list [{PCA,IForest,OC-SVM,AE,Deep-SVDD,USAD,DAGMM,LSTM-VAE} ...]
                        The list of models to evaluate.
  --save-dir SAVE_DIR   The folder to store the model outputs.
  --data-dir DATA_DIR   The folder where the data are stored.
  --is-trained          If the models are already trained. Default action is false.

```

## Example

To train an USAD model with the default parameters with an anomaly contamination of 5%, a window size of 10 and a threshold of 0.8 ($\alpha$ in the paper) an evaluate with WAD metric run :

```bash
python main.py --model-name USAD --contamination-ratio 0.05 --window-size 10 --threshold 0.8 --metric window_wad
```

## Directory structure

```bash
.
├── __init__.py
├── LICENSE                                   
├── main.py                                   
├── README.md
├── requirements.txt
├── setup.py
└── src
    ├── experiments
    │   ├── data_contamination.py
    │   └── metric_quality.py
    ├── __init__.py
    ├── models
    │   ├── anomaly_pca.py
    │   ├── auto_encoder.py
    │   ├── auto_encoder_utils.py
    │   ├── beta_detector.py
    │   ├── dagmm.py
    │   ├── dagmm_utils.py
    │   ├── deep_svd.py
    │   ├── deep_svd_utils.py
    │   ├── iforest.py
    │   ├── __init__.py
    │   ├── lstm_vae.py
    │   ├── lstm_vae_utils.py
    │   ├── oc_svm.py
    │   ├── random_guessing.py
    │   ├── usad.py
    │   ├── usad_utils.py
    │   └── zero_rule.py
    └── utils
        ├── algorithm_utils.py
        ├── data_processing.py
        ├── evaluation_utils.py
        ├── window.py
        └── __init__.py
```
