import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from collections import Counter
from utils.nlp_utils import *

class Client:
    def __init__(self, args, client_id, train_dataset, test_dataset, train_indices, val_indices, test_indices):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.batch_size = args.local_bs
        self.train_dataloader = self.create_dataloader("train")
        self.val_dataloader = self.create_dataloader("val")
        self.test_dataloader = self.create_dataloader("test")
    def get_class_distribution(self, indices, dataset):
        targets = [dataset.targets[idx] for idx in indices]
        return dict(Counter(targets))

    def get_distributions(self):
        train_dist = self.get_class_distribution(self.train_indices, self.train_dataset)
        val_dist = self.get_class_distribution(self.val_indices, self.train_dataset)
        test_dist = self.get_class_distribution(self.test_indices, self.test_dataset)

        return {
            'train': train_dist,
            'val': val_dist,
            'test': test_dist
        }
    
    def create_dataloader(self, dataset_type):
        dataset_dict = {
            "train": (self.train_dataset, self.train_indices),
            "val": (self.train_dataset, self.val_indices),
            "test": (self.test_dataset, self.test_indices)
        }

        dataset, indices = dataset_dict[dataset_type]
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def print_class_distribution(self):
        def get_class_distribution(indices, dataset):
            targets = [dataset.targets[idx] for idx in indices]
            return dict(Counter(targets))

        train_dist = get_class_distribution(self.train_indices, self.train_dataset)
        val_dist = get_class_distribution(self.val_indices, self.train_dataset)
        test_dist = get_class_distribution(self.test_indices, self.test_dataset)

        print(f"Client {self.client_id} class distribution:")
        print(f"  Train: {train_dist}")
        print(f"  Val: {val_dist}")
        print(f"  Test: {test_dist}")
        
    def print_class_distribution(self):
        def get_class_distribution(indices, dataset):
            targets = [dataset.targets[idx] for idx in indices]
            return dict(Counter(targets))

        train_dist = get_class_distribution(self.train_indices, self.train_dataset)
        val_dist = get_class_distribution(self.val_indices, self.train_dataset)
        test_dist = get_class_distribution(self.test_indices, self.test_dataset)

        print(f"Client {self.client_id} class distribution:")
        print(f"  Train: {train_dist}")
        print(f"  Val: {val_dist}")
        print(f"  Test: {test_dist}")

    def check_indices(self):
        def has_duplicates(lst):
            return len(lst) != len(set(lst))

        if has_duplicates(self.train_indices):
            raise ValueError("Duplicate entries found in train_indices")
        if has_duplicates(self.val_indices):
            raise ValueError("Duplicate entries found in val_indices")
        if has_duplicates(self.test_indices):
            raise ValueError("Duplicate entries found in test_indices")

        train_indices_set = set(self.train_indices)
        val_indices_set = set(self.val_indices)

        if not train_indices_set.isdisjoint(val_indices_set):
            raise ValueError("Overlap found between train_indices and val_indices")
        if not val_indices_set.isdisjoint(train_indices_set):
            raise ValueError("Overlap found between val_indices and train_indices")

    def train(self, model, criterion, optimizer, args):
        self.train_dataloader = self.create_dataloader('train')  # Recreate dataloader to shuffle data

        model.train()
        step_count = 0  # Initialize step counter
        while step_count < args.local_ep:  # Loop until local steps are reached
            for inputs, labels in self.train_dataloader:
                if args.device == 'cuda':
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step_count += 1
                if step_count >= args.local_ep:  # Exit if local steps are reached
                    break
        return model

    def inference(self, model, criterion, args, loader_type='test'):
        model.eval()
        correct, total, test_loss = 0.0, 0.0, 0.0
        testloader = self.test_dataloader if loader_type == 'test' else self.val_dataloader
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(testloader):
                if args.device == 'cuda':
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(testloader)
        accuracy = correct / total
        return accuracy, test_loss

    def single_batch_inference(self, model, criterion, args): # Ahmad his function is not necessary atm
        model.eval()
        testloader = iter(self.test_dataloader)
        with torch.no_grad():
            inputs, labels = next(testloader)
            if args.device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(labels)
        return accuracy, loss.item()

class ShakespeareClient:
    def __init__(self, args, client_id, train_subset,test_subset):
        '''
        Initializes a ShakespeareClient instance.

        Parameters:
        args (Namespace): Configuration arguments containing various.
        client_id (str): Identifier for the client.
        train_dataset (Dataset): Subset of the dataset assigned to the client.
        test_subset (Dataset): Subset of the dataset assigned to the client.
        batch_size (int): The size of each batch for data loading (default is 64).
        '''
        self.client_id = client_id
        self.train_dataset = train_subset
        self.test_dataset = test_subset
        self.train_indices = list(range(len(train_subset)))
        self.test_indices = list(range(len(test_subset)))
        self.batch_size = args.local_bs
        self.args = args
        self.train_dataloader = self.create_dataloader('train')
        self.test_dataloader = self.create_dataloader('test')
    def get_class_distribution(self, indices, dataset):
        targets = [dataset.targets[idx] for idx in indices]
        return dict(Counter(targets))

    def get_distributions(self):
        train_dist = self.get_class_distribution(self.train_indices, self.train_dataset)
        test_dist = self.get_class_distribution(self.test_indices, self.test_dataset)

        return {
            'train': train_dist,
            'test': test_dist
        }
    def print_class_distribution(self):
        def get_class_distribution(indices, dataset):
            targets = [dataset.targets[idx] for idx in indices]
            return dict(Counter(targets))

        train_dist = get_class_distribution(self.train_indices, self.train_dataset)
        test_dist = get_class_distribution(self.test_indices, self.test_dataset)

        print(f"Client {self.client_id} class distribution:")
        print(f"  Train: {train_dist}")
        print(f"  Test: {test_dist}")
        
    def print_class_distribution(self):
        def get_class_distribution(indices, dataset):
            targets = [dataset.targets[idx] for idx in indices]
            return dict(Counter(targets))

        train_dist = get_class_distribution(self.train_indices, self.train_dataset)
        test_dist = get_class_distribution(self.test_indices, self.test_dataset)

        print(f"Client {self.client_id} class distribution:")
        print(f"  Train: {train_dist}")
        print(f"  Test: {test_dist}")

    def check_indices(self):
        def has_duplicates(lst):
            return len(lst) != len(set(lst))

        if has_duplicates(self.train_indices):
            raise ValueError("Duplicate entries found in train_indices")
        if has_duplicates(self.test_indices):
            raise ValueError("Duplicate entries found in test_indices")

    def create_dataloader(self, loader_type='train'):
        '''
        Creates a DataLoader for the client's dataset.

        Returns:
        DataLoader: A DataLoader instance for the dataset with specified batch size and shuffling enabled.
        '''
        dataloader = DataLoader(self.train_dataset if loader_type =='train' else self.test_dataset 
                                , batch_size=self.batch_size, shuffle=loader_type =='train')
        return dataloader

    def train(self, model, criterion, optimizer, local_step=4):
        '''
        Trains the model on the client's dataset for a specified number of local steps.

        Parameters:
        model (nn.Module): The model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer for updating model parameters.
        local_step (int): Number of local steps to train (default is 4).

        Returns:
        nn.Module: The trained model.
        '''
        model.train()
        step_count = 0
        while step_count < local_step:
            for inputs, labels in self.train_dataloader:
                labels = labels.squeeze(1)
                if self.args.device == 'cuda':
                    inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
                optimizer.zero_grad()
                hidden = model.init_hidden(inputs.size(0))  # Initialize hidden state for current batch size
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step_count += 1
                if step_count >= local_step:
                    break
        return model

    def inference(self, model, criterion):
        '''
        Performs inference on the client's dataset to evaluate the model.

        Parameters:
        model (nn.Module): The model to be evaluated.
        criterion (nn.Module): The loss function.

        Returns:
        tuple: Accuracy and average loss over the dataset.
        '''
        model.eval()
        correct, total, test_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                labels = labels.squeeze(1)
                if self.args.device == 'cuda':
                    inputs, labels = inputs.cuda(), labels.cuda()
                hidden = model.init_hidden(inputs.size(0))
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(self.train_dataloader)
        accuracy = correct / total
        return accuracy, test_loss

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, show_loading=False, crop_amount=2000, tst_ratio=5, rand_seed=0):
        '''
        Initializes a ShakespeareObjectCrop instance.

        Parameters:
        data_path (str): Path to the directory containing training and test data.
        dataset_prefix (str): Prefix to be used for the dataset name.
        show_loading (bool): Flag to show loading information (default is False).
        crop_amount (int): Amount of data to crop for each client (default is 2000).
        tst_ratio (int): Ratio of test data to crop amount (default is 5).
        rand_seed (int): Random seed for reproducibility (default is 0).
        '''
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path + 'train/', data_path + 'test/')
        self.users = users
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        tst_data_count = 0

        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            if show_loading:
                print(clnt)
                print(len(train_data[users[clnt]]['x']))
                print(crop_amount)
            start = np.random.randint(len(train_data[users[clnt]]['x']) - crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start + crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start + crop_amount]

        tst_data_count = (crop_amount // tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount // tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x']) - curr_amount)
            self.tst_x[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[clnt]]['x'])[start:start + curr_amount]
            self.tst_y[tst_data_count: tst_data_count + curr_amount] = np.asarray(test_data[users[clnt]]['y'])[start:start + curr_amount]
            tst_data_count += curr_amount

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

        # Convert characters to numbers
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)
        self.tst_x_char = np.copy(self.tst_x)
        self.tst_y_char = np.copy(self.tst_y)
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)
            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)

        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        self.tst_x = list(range(len(self.tst_x_char)))
        self.tst_y = list(range(len(self.tst_x_char)))

        for idx in range(len(self.tst_x_char)):
            self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
            self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        self.tst_x = np.asarray(self.tst_x)
        self.tst_y = np.asarray(self.tst_y)

class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, tst_ratio=5, rand_seed=0):
        '''
        Initializes a ShakespeareObjectCrop_noniid instance.

        Parameters:
        data_path (str): Path to the directory containing training and test data.
        dataset_prefix (str): Prefix to be used for the dataset name.
        n_client (int): Number of clients to sample (default is 100).
        crop_amount (int): Amount of data to crop for each client (default is 2000).
        tst_ratio (int): Ratio of test data to crop amount (default is 5).
        rand_seed (int): Random seed for reproducibility (default is 0).
        '''
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path + 'train/', data_path + 'test/')
        self.users = users
        tst_data_count_per_clnt = (crop_amount // tst_ratio)

        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount
                    and len(np.asarray(test_data[users[clnt]]['y'])) > tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]

def cifar_iid(args, train_dataset, test_dataset):
    val_split = args.val_split
    num_clients = args.num_users

    # Number of classes in the dataset
    num_classes = len(train_dataset.classes)

    # Create a list to store indices for each class
    class_indices = [[] for _ in range(num_classes)]

    # Populate class_indices with the indices of each class
    for idx, target in enumerate(train_dataset.targets):
        class_indices[target].append(idx)

    # Shuffle indices within each class
    for indices in class_indices:
        np.random.shuffle(indices)

    # Create lists for train and validation class indices
    train_class_indices = [[] for _ in range(num_classes)]
    val_class_indices = [[] for _ in range(num_classes)]

    # Split the indices into 80% for train and 20% for validation
    for i, indices in enumerate(class_indices):
        split_idx = int(len(indices) * val_split)
        val_class_indices[i] = indices[:split_idx]
        train_class_indices[i] = indices[split_idx:]

    # Prepare test_class_indices
    test_class_indices = [[] for _ in range(num_classes)]
    for idx, target in enumerate(test_dataset.targets):
        test_class_indices[target].append(idx)
    for indices in test_class_indices:
        np.random.shuffle(indices)

    # Calculate the number of samples per client per class
    train_samples_per_client_per_class = int(len(train_dataset) * (1-val_split) // (num_clients * num_classes))
    val_samples_per_client_per_class = int(len(train_dataset) * val_split // (num_clients * num_classes))
    test_samples_per_client_per_class = len(test_dataset) // (num_clients * num_classes)

    # Initialize the list of client objects
    clients = []

    # Distribute the samples uniformly to the clients
    for client_id in range(num_clients):
        train_client_indices = []
        val_client_indices = []
        test_client_indices = []

        for train_class_indices_for_class in train_class_indices:
            train_client_indices.extend(train_class_indices_for_class[client_id * train_samples_per_client_per_class : (client_id + 1) * train_samples_per_client_per_class])
        for val_class_indices_for_class in val_class_indices:
            val_client_indices.extend(val_class_indices_for_class[client_id * val_samples_per_client_per_class : (client_id + 1) * val_samples_per_client_per_class])
        for test_class_indices_for_class in test_class_indices:
            test_client_indices.extend(test_class_indices_for_class[client_id * test_samples_per_client_per_class : (client_id + 1) * test_samples_per_client_per_class])

        client = Client(args, client_id, train_dataset, test_dataset, train_client_indices, val_client_indices, test_client_indices)
        clients.append(client)

    return clients

def cifar_noniid(args, train_dataset, test_dataset):
    def class_clients_sharding(num_classes, Nc):
        class_clients = {key: set() for key in range(num_classes)}
        first_clients = list(range(num_classes))
        clients_list = [num // (Nc-1) for num in range((Nc-1)*100)]
        random.shuffle(first_clients)
        for i in range(num_classes):
            class_clients[i].add(first_clients[i])

        for j in range(1,Nc):
            class_list = list(range(num_classes))
            for i in range(num_classes):
                random_class = random.choice(class_list)
                class_list.remove(random_class)

                clients_list_cleaned = [client for client in clients_list if client not in class_clients[random_class]]

                random_client = random.choice(clients_list_cleaned)
                class_clients[random_class].add(random_client)
                clients_list.remove(random_client)

        return class_clients

    val_split = args.val_split
    num_clients = args.num_users
    Nc = args.Nc
    num_classes = len(train_dataset.classes)

    error = True
    while error:
        try:
            class_clients = class_clients_sharding(num_classes, Nc)
            error = False
        except Exception as e:
            print("Sharding Invalid, trying again...")

    # Create a list to store indices for each class
    class_indices = [[] for _ in range(num_classes)]

    # Populate class_indices with the indices of each class
    for idx, target in enumerate(train_dataset.targets):
        class_indices[target].append(idx)

    # Shuffle indices within each class
    for indices in class_indices:
        np.random.shuffle(indices)

    # Create lists for train and validation class indices
    train_class_indices = [[] for _ in range(num_classes)]
    val_class_indices = [[] for _ in range(num_classes)]

    # Split the indices into 80% for train and 20% for validation
    for i, indices in enumerate(class_indices):
        split_idx = int(len(indices) * val_split)
        val_class_indices[i] = indices[:split_idx]
        train_class_indices[i] = indices[split_idx:]

    # Prepare test_class_indices
    test_class_indices = [[] for _ in range(num_classes)]
    for idx, target in enumerate(test_dataset.targets):
        test_class_indices[target].append(idx)
    for indices in test_class_indices:
        np.random.shuffle(indices)

    # Initialize the list of client objects
    clients_list = []

    # Calculate the number of samples per client per class
    train_samples_per_client_per_class = int(len(train_dataset) * (1-val_split) // (Nc * num_classes))
    val_samples_per_client_per_class = int(len(train_dataset) * val_split // (Nc * num_classes))
    test_samples_per_client_per_class = len(test_dataset) // (Nc * num_classes)


    train_shards_indices = [[] for clients in range(num_clients)]
    val_shards_indices = [[] for clients in range(num_clients)]
    test_shards_indices = [[] for clients in range(num_clients)]

    for class_idx in range(num_classes):
        train_class_indices_for_class = train_class_indices[class_idx]
        val_class_indices_for_class = val_class_indices[class_idx]
        test_class_indices_for_class = test_class_indices[class_idx]
        clients = class_clients[class_idx].copy()
        for client_idx in range(Nc):
            client = random.choice(list(clients))
            clients.remove(client)

            train_start_idx = client_idx * int(train_samples_per_client_per_class)
            val_start_idx = client_idx * int(val_samples_per_client_per_class)
            test_start_idx = client_idx * int(test_samples_per_client_per_class)

            train_end_idx = (client_idx + 1) * int(train_samples_per_client_per_class)
            val_end_idx = (client_idx + 1) * int(val_samples_per_client_per_class)
            test_end_idx = (client_idx + 1) * int(test_samples_per_client_per_class)

            train_shards_indices[client].extend(train_class_indices_for_class[train_start_idx:train_end_idx])
            val_shards_indices[client].extend(val_class_indices_for_class[val_start_idx:val_end_idx])
            test_shards_indices[client].extend(test_class_indices_for_class[test_start_idx:test_end_idx])

    for client_id in range(num_clients):
        client = Client(args, client_id, train_dataset, test_dataset, train_shards_indices[client_id], val_shards_indices[client_id], test_shards_indices[client_id])
        clients_list.append(client)

    return clients_list



