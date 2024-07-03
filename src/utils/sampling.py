import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class Client:
    def __init__(self, client_id, train_dataset, indices, batch_size=64):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.indices = indices
        self.batch_size = batch_size
        self.train_dataloader = self.create_dataloader()

    def create_dataloader(self):
        subset = Subset(self.train_dataset, self.indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self, model, criterion, optimizer, local_steps=4):
        self.train_dataloader = self.create_dataloader()  # Recreate dataloader to shuffle data

        model.train()
        step_count = 0  # Initialize step counter
        while step_count < local_steps:  # Loop until local steps are reached
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                step_count += 1
                if step_count >= local_steps:  # Exit if local steps are reached
                    break
        return model


def cifar_iid(dataset, num_clients):
    # Number of classes in the dataset
    num_classes = len(dataset.dataset.classes)

    # Create a list to store indices for each class
    class_indices = [[] for _ in range(num_classes)]

    # Populate class_indices with the indices of each class
    for idx, target in enumerate(dataset.dataset.targets):
        class_indices[target].append(idx)

    # Shuffle indices within each class
    for indices in class_indices:
        np.random.shuffle(indices)

    # Calculate the number of samples per client per class
    samples_per_client_per_class = len(dataset) // (num_clients * num_classes)
    # Initialize the list of shards
    # a shard is the portion of the dataset belonging to one of the client
    # we separate each shard in 2 portions:
    # - one will be the actual subset of the dataset used for training
    # - the other will be used to create the validation dataset
    train_shards_indices = [[] for clients in range(num_clients)]


    # Distribute the samples uniformly to the clients
    for class_idx in range(num_classes):
        class_indices_for_class = class_indices[class_idx]

        for client_idx in range(num_clients):
            start_idx = client_idx * int(samples_per_client_per_class)
            end_idx = (client_idx + 1) * int(samples_per_client_per_class)
            train_shards_indices[client_idx].extend(class_indices_for_class[start_idx:end_idx])

    # Create subsets for each client
    return [Client(client_id, train_dataset=client_data, indices=range(len(client_data))) for client_id, client_data in enumerate(class_idx)]




def cifar_noniid(dataset, num_clients, Nc):
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

    num_classes = len(dataset.dataset.classes)

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
    for idx, target in enumerate(dataset.dataset.targets):
        class_indices[target].append(idx)

    # Shuffle indices within each class
    for indices in class_indices:
        np.random.shuffle(indices)

    train_shards_indices = [[] for clients in range(num_clients)]
    samples_per_client_per_class = len(dataset) // (Nc * num_classes)

    # Distribute the samples uniformly to the clients
    for class_idx in range(num_classes):
        class_indices_for_class = class_indices[class_idx]
        clients = class_clients[class_idx].copy()
        for client_idx in range(Nc):
        #for client in class_clients[class_idx]:
            client = random.choice(list(clients))
            clients.remove(client)

            start_idx = client_idx * int(samples_per_client_per_class)
            end_idx = (client_idx + 1) * int(samples_per_client_per_class)
            train_shards_indices[client].extend(class_indices_for_class[start_idx:end_idx])

    # Create subsets for each client
    return [Client(client_id, train_dataset=client_data, indices=range(len(client_data))) for client_id, client_data in enumerate(class_idx)]


__all__ = ['cifar_iid', 'cifar_noniid']
