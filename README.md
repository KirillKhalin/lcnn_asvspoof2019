# LCNN Voice Anti-Spoofing (ASVspoof 2019)

<p align="center">
  <a href="#about">About</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#installation">Installation</a> •
  <a href="#important">Important</a> •
  <a href="#usage">Usage</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/pytorch/pytorch">
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
</a>
<a href="https://hydra.cc/">
   <img src="https://img.shields.io/badge/Hydra-1.3-blue">
</a>
<a href="LICENSE">
   <img src="https://img.shields.io/badge/license-MIT-blue.svg">
</a>
</p>

## Project Structure
The root of the project contains directories/files:
  ```text
  lcnn_asvspoof2019/
  ├── src/   (here code of model, utils, loss and etc.)
  ├── .flake8
  ├── .gitignore
  ├── .pre-commit-config.yaml
  ├── CITATION.cff
  ├── LICENSE
  ├── README.md
  ├── inference.py
  ├── requirements.txt
  └── train.py
  ```

## About

This repository contains an implementation of Light CNN (LCNN) for voice anti-spoofing based on the **ASVspoof 2019 Logical Access (LA)** evaluation setup.

The codebase is built using the PyTorch project template, providing configuration management via [Hydra](https://hydra.cc/).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KirillKhalin/lcnn_asvspoof2019.git
   cd lcnn_asvspoof2019
    ```
  
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Important

You need to download the **ASVspoof 2019 (LA)** dataset (for example, [Kaggle link](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)) and update the protocol and data paths to point to your local dataset directory.

### 1. Dataset Paths
You can set your dataset paths directly in the Hydra configuration files:
* `src/configs/datasets/asvspoof.yaml` (for training/validation)
* `src/configs/datasets/asvspoof_eval.yaml` (for evaluation)

Alternatively, you can override them with the command line when running scripts (see examples below).

### 2. Checkpoints and Pre-trained Models
Similarly, you can configure paths for saving and loading model checkpoints:

* **Saving Checkpoints:** By default, Hydra automatically creates a unique directory for each run under `outputs/` (or `saved/`), where training logs and model checkpoints (`.pth`) will be saved. You can override the output directory with CLI:
  ```bash
  python train.py hydra.run.dir=your/custom/save_path
  ```
* **Loading Pre-trained Weights (Inference):**
To specify a pre-trained model for evaluation, you need to override the from_pretrained parameter inside the inferencer config (`src/configs/inference.yaml`).
Alternatively, you can override it when running scripts:
```bash
python inference.py inferencer.from_pretrained=/path/to/your/model_best.pth
```

## Usage

### 1. Training
To run model training with the default Hydra configuration:
```bash
python train.py
```
To change some parameters do, for example:
```bash
python train.py trainer.train_one_batch=True trainer.seed=10
```

### 2. Inferencing
To run model inferencing with the default Hydra configuration:
```bash
python inference.py
```
To change some parameters do, for example:
```bash
python inference.py inferencer.seed = 1000
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
