from utils import *
from utils.options import *
from utils.get_functions import *
from utils.update import *
from utils.sampling import *
from utils.exp_details import *
from utils.average_weights import *
import os
import copy
import numpy as np
from tqdm import tqdm
from models import *
import wandb
import glob
import torch


if __name__ == '__main__':
    args = args_parser()
    #todo: add logger
    if args.wandb_key:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_username)
        wandb.config.update(args)
    else:
        print("No wandb key provided")

    if args.gpu:
        d = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        print(f"Using {d} device")
        torch.cuda.set_device(d)
    device = 'cuda' if args.gpu else 'cpu'
    
    # Transformation for CIFAR-100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Download and load the training dataset
    trainset_full = datasets.CIFAR100(root='./data', train=True, download=True)

    # Split the dataset into train and validation sets
    num_train = int(len(trainset_full) * 0.8)  # 80% of data for training
    num_val = len(trainset_full) - num_train   # 20% of data for validation
    generator1 = torch.Generator().manual_seed(42)
    trainset, valset = random_split(trainset_full, [num_train, num_val], generator=generator1)

    # Apply the correct transforms to the subsets
    trainset = Subset(datasets.CIFAR100(root='./data', train=True, transform=transform_train), trainset.indices)
    valset = Subset(datasets.CIFAR100(root='./data', train=True, transform=transform_test), valset.indices)

    # DataLoader for the training and validation sets
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    # Download and load the test dataset
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)