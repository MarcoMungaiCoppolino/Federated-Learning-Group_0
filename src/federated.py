from utils.options import args_parser
from utils.logger import Logger
import torch
from utils.data_utils import get_dataset
from utils.wandb_utils import WandbLogger
import pandas as pd
import os
from models import *
from utils.algorithms import fedAVG

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
    train_set, test_set, user_groups_train = get_dataset(args)


    logger.info("######################")
    logger.info("### Configuration ####")
    logger.info("######################")
    logger.info("Dataset: {}".format(args.dataset))
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
    metrics = pd.DataFrame(columns=['Round', 'Test Accuracy', 'Test Loss', 'Avg Train Accuracy', 'Avg Train Loss'])
    if args.gpu is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == 'cifar':
        global_model = CIFARLeNet().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        for client in user_groups_train:
          client.print_class_distribution()
        fedAVG(global_model, user_groups_train, criterion, args, logger, metrics, wandb_logger, device, test_set)