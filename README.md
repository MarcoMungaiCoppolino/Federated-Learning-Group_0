# Federated Learning Project

This project implements a federated learning framework for training models across multiple clients in a distributed and secure manner. The code leverages [Weights and Biases (WandB)](https://wandb.ai/) for logging, and allows customization of various hyperparameters via command-line arguments or notebook configuration.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running in a Script](#running-in-a-script)
  - [Running in a Jupyter Notebook](#running-in-a-jupyter-notebook)
- [Arguments](#arguments)
- [Project Structure](#project-structure)

## Installation

### Requirements

- Python 3.x
- `pytorch`
- `faiss-gpu` (for fast similarity search in dense vectors)
- WandB for logging

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/MarcoMungaiCoppolino/Federated-Learning-Group_0.git
    cd Federated-Learning-Group_0
    ```

2. Install dependencies using Conda:

    ```bash
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install matplotlib numpy pandas
    pip install -r requirements.txt
    ```

3. Optionally, mount Google Drive (in Colab):

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Usage

### Running in a Script

To run the federated learning pipeline from the command line, use the following command:

```bash
python src/federated.py --wandb_key <YOUR_WANDB_KEY> --epochs 1000 --num_users 50
```

You can pass various arguments to customize the training (see the [Arguments](#arguments) section below).

### Running in a Jupyter Notebook

In case you're running the project in a Jupyter notebook (like Colab), ensure you have all the necessary libraries installed and your Google Drive mounted. Follow these steps:

1. Clone the repository and install the required libraries as described above.
2. Configure WandB credentials:
    ```python
    wandb_key = 'your_wandb_key'
    wandb_username = 'your_wandb_username'
    ```

3. Run the training code cells to initiate the federated learning process.

## Arguments

Here is a table detailing all the available command-line arguments and their descriptions:

| Argument              | Type    | Default                                    | Description                                                  |
|-----------------------|---------|--------------------------------------------|--------------------------------------------------------------|
| `--wandb_key`         | `str`   | `''`                                       | WandB API key for logging                                     |
| `--wandb_username`    | `str`   | `''`                                       | WandB username                                                |
| `--wandb_project`     | `str`   | `'federated_learning'`                     | WandB project name                                            |
| `--wandb_run_name`    | `str`   | `'federated_learning_uniform'`             | WandB run name                                                |
| `--logfile`           | `str`   | `'/content/logger.log'`                    | Log file name                                                 |
| `--data_dir`          | `str`   | `'/content/drive/MyDrive/MLDL/cifar/data'` | Data directory                                                |
| `--epochs`            | `int`   | `2000`                                     | Number of rounds of training                                  |
| `--num_users`         | `int`   | `100`                                      | Number of users: K                                            |
| `--Nc`                | `int`   | `5`                                        | Number of classes each client holds in non-IID setting        |
| `--frac`              | `float` | `0.1`                                      | Fraction of clients participating per round: C                |
| `--val_split`         | `float` | `0.2`                                      | Validation split                                              |
| `--local_ep`          | `int`   | `4`                                        | Number of local epochs: J                                     |
| `--local_bs`          | `int`   | `64`                                       | Local batch size: B                                           |
| `--lr`                | `float` | `0.01`                                     | Learning rate                                                 |
| `--algorithm`         | `str`   | `'fedavg'`                                 | Federated learning algorithm to use                           |
| `--inner_wd`          | `float` | `4e-3`                                     | Inner weight decay                                            |
| `--wd`                | `float` | `4e-4`                                     | Weight decay                                                  |
| `--model`             | `str`   | `'cnn'`                                    | Model type (e.g., 'cnn', 'mlp')                               |
| `--optimizer`         | `str`   | `'sgd'`                                    | Optimizer type (e.g., 'sgd', 'adam')                          |
| `--dataset`           | `str`   | `'cifar'`                                  | Dataset name                                                  |
| `--num_classes`       | `int`   | `100`                                      | Number of classes in the dataset                              |
| `--iid`               | `int`   | `0`                                        | Whether to use IID data split (1 for IID, 0 for non-IID)      |
| `--participation`     | `int`   | `1`                                        | Uniform client participation (1 for uniform, 0 for skewed)    |
| `--backup`            | `int`   | `1900`                                     | Frequency of old backup preservation                          |
| `--checkpoint_path`   | `str`   | `'.'`                                      | Directory for saving checkpoints                              |
| `--print_every`       | `int`   | `10`                                       | Frequency of printing train accuracy and saving checkpoints   |
| `--verbose`           | `int`   | `0`                                        | Verbosity level (0 for silent, 1 for verbose)                 |
| `--gpu`               | `None`  | `None`                                     | GPU ID to use (default is CPU)                                |

## Project Structure

- **`main_runner.ipynb`**: Jupyter notebook for running the main federated learning experiment.
- **`requirements.txt`**: A list of required libraries for setting up the environment.
- **`scripts/`**: Contains various scripts for different datasets and tasks.
  - `README.md`: Documentation related to scripts.
  - `cifar/`: Scripts and resources specific to the CIFAR dataset applying fedavg.
  - `personalised/`: Scripts for personalized federated learning applying generalization on fedavg and pfedhn.
  - `shakespeare/`: Scripts related to the Shakespeare dataset applying fedavg.
- **`src/`**: Source code directory containing the core logic.
  - `federated.py`: The main script for running federated learning experiments.
  - `models/`: Contains model definitions (e.g., CNN, MLP).
  - `utils/`: Utility functions for data processing, logging, and more.
- **`notebooks/`**: Contains additional Jupyter notebooks for various experiments.
  - `Centralized_baseline_CIFAR100.ipynb`: Baseline experiment on the CIFAR-100 dataset.
  - `Centralized_baseline_Shakespeare.ipynb`: Baseline experiment on the Shakespeare dataset.
  - `Knn_per.ipynb`: Notebook for k-Nearest Neighbors-PER experiments.
  - `plots.ipynb`: Notebook for generating plots and visualizations.
- **`LICENSE`**: Licensing information for the project.
- **`README.md`**: Documentation for the project.

