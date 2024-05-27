import torch
from torchvision import datasets, transforms
from .sampling import *

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups_train, user_groups_val, user_groups_test = cifar_iid(train_dataset, args.num_users)
        else:
            user_groups_train, user_groups_val, user_groups_test = cifar_noniid(train_dataset, args.num_users, args.Nc)

    elif args.dataset == 'shakespears':
        print("ancora da finire")

    return train_dataset, test_dataset, user_groups_train, user_groups_val, user_groups_test


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
        participation = input("Enter Participation parameter (1 for Uniform, 0 for Skewed): ")
        if participation in ['1', '0']:
            participation = int(participation)
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

    return iid, participation, Nc, J


__all__ = ["get_dataset", "get_user_input"]
