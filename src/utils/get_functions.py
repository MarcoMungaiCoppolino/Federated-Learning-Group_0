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


def get_user_input():
    print("Modify Parameters")

    # Modify iid parameter
    while True:
        iid = input("Enter IID parameter (1 for IID, 0 for non-IID): ")
        if iid in ['1', '0']:
            iid = int(iid)
            break
        else:
            print("Invalid input. Please enter 1 or 0.")

    # Modify participation parameter
    while True:
        partecipation = input("Enter Participation parameter (1 for Uniform, 0 for Skewed): ")
        if partecipation in ['1', '0']:
            partecipation = int(partecipation)
            break
        else:
            print("Invalid input. Please enter 1 or 0.")

    # If non-IID, ask for Nc parameter
    Nc = 50  # Default value
    if iid == 0:
        while True:
            Nc = input("Enter Nc parameter (1, 5, 10, 50): ")
            if Nc in ['1', '5', '10', '50']:
                Nc = int(Nc)
                break
            else:
                print("Invalid input. Please enter 1, 5, 10, or 50.")

    # Initialize J parameter
    J = 4

    # If non-IID, ask for J parameter
    if iid == 0:
        while True:
            J = input("Enter J parameter (4, 8, 16): ")
            if J in ['4', '8', '16']:
                J = int(J)
                break
            else:
                print("Invalid input. Please enter 4, 8, or 16.")

    return iid, partecipation, Nc, J


__all__ = ["get_dataset", "get_user_input"]
