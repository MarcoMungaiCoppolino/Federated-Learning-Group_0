# Federated Learning Experiments with FedAvg and pFedHN

This repository contains scripts to run federated learning experiments using FedAvg and pFedHN on the Shakespeare and CIFAR datasets. The scripts are organized into three main directories: `personalised`, `shakespeare`, and `cifar`.

## Directory Structure

- **personalised**: Scripts to run personalized federated learning algorithms (FedAvg and pFedHN).
  - `fedavg_uniform.sh`: Run FedAvg on a uniformly distributed dataset.
  - `fedavg_skewed.sh`: Run FedAvg on a skewed dataset.
  - `pfedhn_uniform.sh`: Run pFedHN on a uniformly distributed dataset.
  - `pfedhn_skewed.sh`: Run pFedHN on a skewed dataset.
  
- **shakespeare**: Scripts to run federated learning experiments on the Shakespeare dataset.
  - `federated_iid_uniform.sh`: Run an IID (independent and identically distributed) experiment on a uniformly distributed dataset.
  - `federated_iid_skewed.sh`: Run an IID experiment on a skewed dataset.
  - `federated_niid_uniform.sh`: Run a non-IID experiment on a uniformly distributed dataset.
  - `federated_niid_skewed.sh`: Run a non-IID experiment on a skewed dataset.
  - `setup_dataset.sh`: Script to prepare the Shakespeare dataset.

- **cifar**: Scripts to run federated learning experiments on the CIFAR dataset.
  - `federated_iid_uniform.sh`: Run an IID experiment on a uniformly distributed dataset.
  - `federated_iid_skewed.sh`: Run an IID experiment on a skewed dataset.
  - `federated_niid_uniform.sh`: Run a non-IID experiment on a uniformly distributed dataset.
  - `federated_niid_skewed.sh`: Run a non-IID experiment on a skewed dataset.

## Running the Scripts

1. Navigate to the desired dataset directory (`personalised`, `shakespeare`, or `cifar`).
2. Choose the appropriate bash script depending on whether you want to run the experiment on a uniform or skewed dataset, and whether you're using FedAvg or pFedHN.
3. Run the script:
   ```bash
   bash script_name.sh
## Dataset Setup (Shakespeare)
Before running any experiments on the Shakespeare dataset, make sure to prepare the dataset by executing the `setup_dataset.sh` script:
   ```bash
    bash setup_dataset.sh
