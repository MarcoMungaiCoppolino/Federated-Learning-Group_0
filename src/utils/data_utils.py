from torchvision import datasets, transforms
from utils.sampling import *

def get_dataset(args):
    data_dir = args.data_dir
    if args.dataset == 'cifar':
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
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transform_test)
        if args.iid:
            # Sample IID user data from CIFAR100
            user_groups_train = cifar_iid(train_dataset, test_dataset, args.val_split, args.num_users, args)
        else:
            user_groups_train = cifar_noniid(train_dataset, args.num_users, args.Nc, args)
        
        return train_dataset, test_dataset, user_groups_train