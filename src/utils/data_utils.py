from torchvision import datasets, transforms
from utils.sampling import *
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
            clients = cifar_iid(args, train_dataset, test_dataset)
        else:
            clients = cifar_noniid(args, train_dataset, test_dataset)
    else:
        clients=[]
        storage_path = args.data_dir
        if args.iid:
            name = 'shakepeare_iid'
            data_obj = ShakespeareObjectCrop(storage_path, name)
        else:
            name = 'shakepeare_noniid'
            number_of_clients=args.num_users
            data_obj = ShakespeareObjectCrop_noniid(storage_path,name,number_of_clients)
        for (client_id, client_x,client_y, test_x, test_y) in enumerate(zip(data_obj.clnt_x,data_obj.clnt_y,data_obj.tst_x,data_obj.tst_y)):
            #create customDataset for storing client data
            client_dataset = ShakespeareDataset(client_x, client_y)
            client_test_dataset = ShakespeareDataset(test_x, test_y)
            #create client giving as input the local dataset(no indices needed)
            client = ShakespeareClient(args, client_id=client_id, subset=client_dataset, test_subset=client_test_dataset)
            clients.append(client)
        train_dataset = ShakespeareDataset(data_obj.clnt_x, data_obj.clnt_y)
        test_dataset = ShakespeareDataset(data_obj.tst_x, data_obj.tst_y)
    return train_dataset, test_dataset, clients 