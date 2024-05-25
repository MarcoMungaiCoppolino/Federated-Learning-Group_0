import torch
from torchvision import datasets, transforms
from .sampling import *

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        #TODO: @Marco: update the dataset to CIFAR100
        train_set, val_set = stratified_split(train_dataset, val_split=0.2)


        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from cifar
            user_groups = cifar_iid(train_set, args.num_users)
        else:
            user_groups = cifar_noniid(train_set, args.num_users, args.Nc)

    elif args.dataset == 'shakespears':
        #TODO: implement the shakespears dataset
        print("ancora da finire")

    return train_set, val_set, test_dataset, user_groups

__all__ = ['get_dataset']