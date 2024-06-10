import argparse
import sys


def args_parser():
    parser = argparse.ArgumentParser()
    ## Wandb parameters
    parser.add_argument('--wandb_key', type=str, default='', help='wandb key')
    parser.add_argument('--wandb_username', type=str, default='', help='wandb userna,,e')
    parser.add_argument('--wandb_project', type=str, default='charBert', help='wandb project')
    parser.add_argument('--wandb_run_name', type=str, default='charBert', help='wandb run name')
    parser.add_argument('logfile', type=str, default='LOG', help='log file name')
    parser.add_argument('--data_dir', type=str, default='', help='data directory')

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=2000,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--Nc', type=int, default=5,
                        help='number of class each client in non iid')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=4,
                        help="the number of local rounds: J")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--checkpoint_resume', type=int, default=0, help='resume from checkpoint, 0 for False, 1 for True')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=100, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--participation', type=int, default=1, 
                        help='Default set to Uniform Participation. Set to 0 for Skewed')
    parser.add_argument('--backup', type=int, default=500,
                        help='How often an old backup should be preserved')
    parser.add_argument('--checkpoint_path', type=str, default=".",
                        help='Saved models location')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--print_every', type=int, default=10,
                        help='how often the train_accuracy is computed, and \
                        how often a new checkpoint is saved')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--gamma', type=int, default=0.1, help='gamma')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # If running in a notebook, ignore the first argument which is the script name
    args = parser.parse_args(args=sys.argv[1:] if "__file__" in globals() else [])
    return args


__all__ = ['args_parser']
