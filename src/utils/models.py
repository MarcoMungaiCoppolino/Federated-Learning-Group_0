import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import functional as F
from utils.datastore import DataStore
from collections import defaultdict
import torch.nn as nn




def load_checkpoint(filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        print(f"Loading checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None
def knn_inference(global_model, args, net_dataidx_map_train, net_dataidx_map_test, loader_type='test', n_parties=100, logger=None):
        test_results = defaultdict(lambda: defaultdict(list))
        for net_id in range(n_parties):
            
            global_model.eval()
            if loader_type == 'train':
                train_results = defaultdict(lambda: defaultdict(list))
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            n_train_samples = len(dataidxs_train.dataset)
            capacity = int(args.Nc * n_train_samples)
            rng = np.random.default_rng(seed=0)
            # vec_dim = 128*65
            vec_dim = 192
            datastore = DataStore(capacity, "random", vec_dim, rng, logger=logger)
            test_correct, test_total, test_avg_loss = compute_accuracy_loss_knn(global_model, dataidxs_train, dataidxs_test, datastore, vec_dim, args, device=args.device)
            if loader_type == 'train':
                train_correct, train_total, train_avg_loss = compute_accuracy_loss_knn(global_model, dataidxs_train, dataidxs_train, datastore, vec_dim, args, device=args.device)
                train_results[net_id]['accuracy'] = train_correct
                train_results[net_id]['loss'] = train_avg_loss
                train_results[net_id]['correct'] = train_correct
                train_results[net_id]['total'] = train_total
            test_results[net_id]['loss'] = test_avg_loss 
            test_results[net_id]['correct'] = test_correct
            test_results[net_id]['total'] = test_total
        test_total_correct = sum([val['correct'] for val in test_results.values()])
        test_total_samples = sum([val['total'] for val in test_results.values()])
        test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
        test_avg_acc = test_total_correct / test_total_samples

        test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
        if loader_type == 'train':
            train_total_correct = sum([val['correct'] for val in train_results.values()])
            train_total_samples = sum([val['total'] for val in train_results.values()])
            train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
            train_acc_pre = train_total_correct / train_total_samples

            train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
            return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
        else:
            return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc

def compute_accuracy_loss_knn(model, train_dataloader, test_dataloader, datastore, embedding_dim, args, device="cuda"):

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)
    datastore.clear()
    n_samples = len(train_dataloader.dataset)
    total = len(test_dataloader.dataset)

    if type(test_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
        test_dataloader = [test_dataloader]

    n_classes = 100

    with torch.no_grad():
        train_features = 0
        train_labels = 0

        ff = 0
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                t_feature = model.produce_feature(x).detach()
                out = model.fc3(t_feature)
                t_feature = t_feature.cpu().numpy()

                if ff == 0:
                    ff = 1
                    train_labels = target.data.cpu().numpy()
                    train_features = t_feature
                else:
                    train_labels = np.hstack((train_labels, target.data.cpu().numpy()))
                    train_features = np.vstack((train_features, t_feature))

        test_features = 0
        test_labels = 0
        test_outputs = 0
        ff = 0
        for tmp in test_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                t_feature = model.produce_feature(x).detach()
                out = model.fc3(t_feature)
                t_feature = t_feature.cpu().numpy()

                if ff == 0:
                    ff = 1
                    test_labels = target.data.cpu().numpy()
                    test_features = t_feature
                    test_outputs = F.softmax(out, dim=1).cpu().numpy()
                else:
                    test_labels = np.hstack((test_labels, target.data.cpu().numpy()))
                    test_features = np.vstack((test_features, t_feature))
                    test_outputs = np.vstack((test_outputs, F.softmax(out, dim=1).cpu().numpy()))
        datastore.build(train_features, train_labels)
        distances, indices = datastore.index.search(test_features, args.k_value)
        similarities = np.exp(-distances / (embedding_dim * 1.))
        neighbors_labels = datastore.labels[indices]
        masks = np.zeros(((n_classes,) + similarities.shape))
        for class_id in range(n_classes):
            masks[class_id] = neighbors_labels == class_id

        knn_outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)
        knn_outputs = knn_outputs.T
        outputs = args.knn_weight * knn_outputs + (1 - args.knn_weight) * test_outputs

        predictions = np.argmax(outputs, axis=1)
        correct = (test_labels == predictions).sum()

    total_loss = criterion(torch.tensor(outputs), torch.tensor(test_labels))
    return correct, total, total_loss

def update_weights(model, weights):
    assert len(weights) == len(list(model.parameters())), "Number of weights must match number of model parameters"
    for param, weight in zip(model.parameters(), weights):
        param.data.copy_(weight)

def initialize_hidden_state(num_layers, hidden_size, batch_size):
    return (torch.zeros(num_layers, batch_size, hidden_size),
            torch.zeros(num_layers, batch_size, hidden_size))

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def inference(model, test_set, criterion, args):
    model.eval()
    correct, total, test_loss = 0.0, 0.0, 0.0
    testloader = DataLoader(test_set, batch_size=args.local_bs,
                            shuffle=False)
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

def eval_knnper_grid(client_, weights_grid_, capacities_grid_):
    client_results = np.zeros((len(weights_grid_), len(capacities_grid_)))

    for ii, capacity in enumerate(capacities_grid_):
        client_.capacity = capacity
        client_.clear_datastore()
        client_.build_datastore()
        client_.gather_knn_outputs()

        for jj, weight in enumerate(weights_grid_):
            client_results[jj, ii] = client_.evaluate(weight) * client_.n_test_samples

    return client_results
__all__ = ['knn_inference','initialize_hidden_state', 'update_weights', 'load_checkpoint','inference', 'save_checkpoint', 'compute_accuracy_loss_knn']