import random
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader, Subset
from utils.models import *
from models import CIFAR_LENET
from utils.datastore import DataStore
import warnings



class Client:
    def __init__(self, args, client_id, train_dataset, test_dataset, train_indices, val_indices, test_indices, logger, model=None):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.logger = logger
        self.device = args.device
        self.batch_size = args.local_bs
        self.train_dataloader = self.create_dataloader("train")
        self.val_dataloader = self.create_dataloader("val")
        self.test_dataloader = self.create_dataloader("test")  
        self.n_train_samples = len(self.train_dataloader.dataset)
        self.n_test_samples = len(self.test_dataloader.dataset)
        self.n_val_samples = len(self.val_dataloader.dataset)
        self.features_dimension = 192
        self.train_knn_outputs_flag = False
        self.num_classes = 100
        self.capacity = -1
        self.strategy = "random"
        self.rng = np.random.default_rng(seed=0)
        self.model = model
        self.datastore = DataStore(self.capacity, self.strategy, self.features_dimension, self.rng)
        self.datastore_flag = False
        self.features_flag = False
        self.train_features = np.zeros(shape=(self.n_train_samples, self.features_dimension), dtype=np.float32)
        self.train_labels = np.zeros(shape=self.n_train_samples, dtype=np.float32)
        self.test_features = np.zeros(shape=(self.n_test_samples, self.features_dimension), dtype=np.float32)
        self.test_labels = np.zeros(shape=self.n_test_samples, dtype=np.float32)
        self.train_model_outputs = np.zeros(shape=(self.n_train_samples, self.num_classes), dtype=np.float32)
        self.train_model_outputs_flag = False
        self.test_model_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.test_model_outputs_flag = False
        self.train_knn_outputs = np.zeros(shape=(self.n_train_samples, self.num_classes), dtype=np.float32)
        self.test_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.test_knn_outputs_flag = False

    def gather_knn_outputs(self, mode="test", scale=1.):
        """
        computes the k-NN predictions

        :param mode: possible are "train" and "test", default is "test"
        :param scale: scale of the gaussian kernel, default is 1.0
        """
        if self.capacity <= 0:
            warnings.warn("trying to gather knn outputs with empty datastore", RuntimeWarning)
            return

        assert self.features_flag, "Features should be computed before building datastore!"
        assert self.datastore_flag, "Should build datastore before computing knn outputs!"

        if mode == "train":
            features = self.train_features
            self.train_knn_outputs_flag = True
        else:
            features = self.test_features
            self.test_knn_outputs_flag = True

        distances, indices = self.datastore.index.search(features, self.k)
        similarities = np.exp(-distances / (self.features_dimension * scale))
        neighbors_labels = self.datastore.labels[indices]

        masks = np.zeros(((self.num_classes,) + similarities.shape))
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        if mode == "train":
            self.train_knn_outputs = outputs.T
        else:
            self.test_knn_outputs = outputs.T
    def build_datastore(self):
        assert self.features_flag, "Features should be computed before building datastore!"
        self.datastore_flag = True
        self.datastore.build(self.train_features, self.train_labels)

    def clear_datastore(self):
        """
        clears `datastore`

        """
        self.datastore.clear()
        self.datastore.capacity = self.capacity

        self.datastore_flag = False
        self.train_knn_outputs_flag = False
        self.test_knn_outputs_flag = False
    def compute_features_and_model_outputs(self):
        """
        extract features from `train_iterator` and `test_iterator` .
        and computes the predictions of the base model (i.e., `self.model`) over `test_iterator`.

        """
        self.features_flag = True
        self.train_model_outputs_flag = True
        self.test_model_outputs_flag = True

        self.train_features, self.train_model_outputs, self.train_labels = \
            compute_embeddings_and_outputs(
                model=self.model,
                iterator=self.train_dataloader,
                embedding_dim=self.features_dimension,
                n_classes=self.num_classes,
                apply_softmax=False,
                device=self.device
            )

        self.test_features, self.test_model_outputs, self.test_labels = \
            compute_embeddings_and_outputs(
                model=self.model,
                iterator=self.test_dataloader,
                embedding_dim=self.features_dimension,
                n_classes=self.num_classes,
                apply_softmax=False,
                device=self.device
            )

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

        self.logger.info(f"Client {self.client_id} class distribution:")
        self.logger.info(f"  Train: {train_dist}")
        self.logger.info(f"  Val: {val_dist}")
        self.logger.info(f"  Test: {test_dist}")

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

    def single_batch_inference(self, model, criterion, args):
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
    
def cifar_iid(args, train_dataset, test_dataset, logger):
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

        client = Client(args, client_id, train_dataset, test_dataset, train_client_indices, val_client_indices, test_client_indices,logger)
        clients.append(client)

    return clients

def cifar_noniid(args, train_dataset, test_dataset, logger, model):
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
        client = Client(args, client_id, train_dataset, test_dataset, train_shards_indices[client_id], val_shards_indices[client_id], test_shards_indices[client_id],logger, model)
        clients_list.append(client)

    return clients_list


__all__ = ['cifar_iid', 'cifar_noniid']
