from utils.options import args_parser
from utils.logger import Logger
import torch
from utils.data_utils import get_dataset
from utils.wandb_utils import WandbLogger
import pandas as pd
import os
from models import *
from utils.algorithms import fedAVG, pFedHN
import random

if __name__ == '__main__':
    args = args_parser()
    wandb_logger = WandbLogger(args)
    logger = Logger("LOG", logfile=args.logfile).logger
    if args.gpu:
        d = f"cuda:{args.gpu}" if args.gpu is not None else ""
        if torch.cuda.is_available():
            torch.cuda.set_device(d)
    device = 'cuda' if args.gpu else 'cpu'
    logger.debug(f"Using {device} device")
    args.device = device
    criterion = nn.CrossEntropyLoss().to(device)

    if args.dataset == 'cifar':
        if args.algorithm == 'fedavg':
            global_model = CIFARLeNet().to(device)
        else:
            global_model = CIFARLeNet().to(device)
    else:
        global_model = CharLSTM().to(device)
    train_set, test_set, clients = get_dataset(args)
    logger.info(test_set)
    logger.info(train_set)
    logger.info("######################")
    logger.info("### Configuration ####")
    logger.info("######################")
    logger.info("Dataset: {}".format(args.dataset))
    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Data Directory: {}".format(args.data_dir))
    logger.info("Checkpoint Directory: {}".format(args.checkpoint_path))
    logger.info(f"Mehod:{'IID' if args.iid else 'Non-IID'} Participation:{'Uniform' if args.participation else 'Skewed'}")
    logger.info("Number of users: {}".format(args.num_users))
    logger.info("Number of classes: {}".format(args.num_classes))
    logger.info("Number of global epochs: {}".format(args.epochs))
    logger.info("Number of local epochs: {}".format(args.local_ep))
    logger.info("Number of local batches: {}".format(args.local_bs))
    logger.info("Learning rate: {}".format(args.lr))
    if args.wandb_key:
        logger.info("Using wandb")
        logger.info(f"Project: {args.wandb_project}")
        logger.info(f"Run name: {args.wandb_run_name}")
        logger.info(f"Running can be found at: {wandb_logger.get_execution_link()}")
    logger.info("######################")
    logger.info("######################")
    if args.dataset == 'cifar':
        metrics = pd.DataFrame(columns=['Round', 'Test Accuracy', 'Test Loss', 'Avg Test Accuracy', 'Avg Test Loss', 'Avg Validation Accuracy', 'Avg Validation Loss'])
    else:
        metrics = pd.DataFrame(columns=['Round', 'Test Accuracy', 'Test Loss', 'Avg Test Accuracy', 'Avg Test Loss'])

    if args.gpu is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == 'cifar':
        clients_classes = {'client_id': [], 'train': [], 'val': [], 'test': [], 'new_id': [], 'train_indices': [], 'val_indices': [], 'test_indices': []}
        new_indices = list(range(100))
        random.shuffle(new_indices)
        for client in clients:
            distributions = client.get_distributions()
            clients_classes['client_id'].append(client.client_id)
            clients_classes['train'].append(distributions['train'])
            clients_classes['test'].append(distributions['test'])
            clients_classes['train_indices'].append(client.train_indices)
            clients_classes['test_indices'].append(client.test_indices)
            clients_classes['val'].append(distributions['val'])
            clients_classes['val_indices'].append(client.val_indices)
            clients_classes['new_id'].append(new_indices[client.client_id])

            
            
        clients_classes_df = pd.DataFrame(clients_classes)
        if args.iid:
            if args.participation:
                pickle_file = f"{args.metrics_dir}/clients_classes_dist_{args.algorithm}_{args.iid}_{args.participation}.pkl"
            else:
                pickle_file = f"{args.metrics_dir}/clients_classes_dist_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}.pkl"
        else:
            if args.participation:
                pickle_file = f"{args.metrics_dir}/clients_classes_dist_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pkl"
            else:
                pickle_file = f"{args.metrics_dir}/clients_classes_dist_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pkl"

        clients_classes_df.to_pickle(pickle_file)
        logger.info(f"Saved clients classes distribution to {pickle_file}")
        # for client in clients:
        #   client.print_class_distribution()
    if args.n_nodes == args.num_users:
        if args.algorithm == 'fedavg':
            fedAVG(global_model, clients, criterion, args, logger, metrics, wandb_logger, device, test_set)
    
