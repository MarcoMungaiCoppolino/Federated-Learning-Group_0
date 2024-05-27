import random
import numpy as np
from torch.utils.data import Subset

def cifar_iid(dataset, num_clients):
    # Number of classes in the dataset
    num_classes = len(dataset.classes)

    # Create a list to store indices for each class
    class_indices = [[] for _ in range(num_classes)]

    # Populate class_indices with the indices of each class
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)

    # Shuffle indices within each class
    for indices in class_indices:
        np.random.shuffle(indices)

    # Calculate the number of samples per client per class
    samples_per_client_per_class = len(dataset) // (num_clients * num_classes)
    
    # Initialize the list of shards
    train_shards_indices = [[] for _ in range(num_clients)]
    val_shards_indices = [[] for _ in range(num_clients)]
    test_shards_indices = [[] for _ in range(num_clients)]

    # Distribute the samples uniformly to the clients
    for class_idx in range(num_classes):
        class_indices_for_class = class_indices[class_idx]
        
        for client_idx in range(num_clients):
            start_idx = client_idx * samples_per_client_per_class
            end_idx = start_idx + samples_per_client_per_class
            
            val_sample_count = max(1, int(0.1 * samples_per_client_per_class))
            test_sample_count = max(1, int(0.1 * samples_per_client_per_class))
            
            # Ensure we don't exceed the samples per client per class
            if val_sample_count + test_sample_count >= samples_per_client_per_class:
                val_sample_count = test_sample_count = 1
            
            end_val_idx = start_idx + val_sample_count
            end_test_idx = end_val_idx + test_sample_count
            end_train_idx = end_idx
            
            train_shards_indices[client_idx].extend(class_indices_for_class[end_test_idx:end_train_idx])
            val_shards_indices[client_idx].extend(class_indices_for_class[start_idx:end_val_idx])
            test_shards_indices[client_idx].extend(class_indices_for_class[end_val_idx:end_test_idx])
            
            
    # Create subsets for each client
    client_trainset = [Subset(dataset, train_shard_indices) for train_shard_indices in train_shards_indices]
    client_valset = [Subset(dataset, val_shard_indices) for val_shard_indices in val_shards_indices]
    client_testset = [Subset(dataset, test_shard_indices) for test_shard_indices in test_shards_indices]

    return client_trainset, client_valset, client_testset


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

    num_classes = len(dataset.classes)

    error = True
    while error:
        try:
            class_clients = class_clients_sharding(num_classes, Nc)
            error = False
        except Exception as e:
            print("Sharding Invalid, trying again...")
            
    for i in range(num_classes):
        print(class_clients[i])
        
    # Create a list to store indices for each class
    class_indices = [[] for _ in range(num_classes)]

    # Populate class_indices with the indices of each class
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)

    # Shuffle indices within each class
    for indices in class_indices:
        np.random.shuffle(indices)

    # Calculate the number of samples per client per class
    samples_per_client_per_class = len(dataset) // (Nc * num_classes)

    # Initialize the list of shards
    train_shards_indices = [[] for _ in range(num_clients)]
    val_shards_indices = [[] for _ in range(num_clients)]
    test_shards_indices = [[] for _ in range(num_clients)]

    # Distribute the samples uniformly to the clients
    for class_idx in range(num_classes):
        class_indices_for_class = class_indices[class_idx]
        clients = class_clients[class_idx].copy()

        for client_idx in range(Nc):
        
            client = random.choice(list(clients))
            clients.remove(client)

            start_idx = client_idx * samples_per_client_per_class
            end_idx = start_idx + samples_per_client_per_class

            val_sample_count = max(1, int(0.1 * samples_per_client_per_class))
            test_sample_count = max(1, int(0.1 * samples_per_client_per_class))

            # Ensure we don't exceed the samples per client per class
            if val_sample_count + test_sample_count >= samples_per_client_per_class:
                val_sample_count = test_sample_count = 1
            
            end_val_idx = start_idx + val_sample_count
            end_test_idx = end_val_idx + test_sample_count
            end_train_idx = end_idx

            train_shards_indices[client].extend(class_indices_for_class[end_test_idx:end_train_idx])
            val_shards_indices[client].extend(class_indices_for_class[start_idx:end_val_idx])
            test_shards_indices[client].extend(class_indices_for_class[end_val_idx:end_test_idx])

    # Create subsets for each client
    client_trainset = [Subset(dataset, train_shard_indices) for train_shard_indices in train_shards_indices]
    client_valset = [Subset(dataset, val_shard_indices) for val_shard_indices in val_shards_indices]
    client_testset = [Subset(dataset, test_shard_indices) for test_shard_indices in test_shards_indices]

    return client_trainset, client_valset, client_testset

__all__ = ['cifar_iid', 'cifar_noniid']
