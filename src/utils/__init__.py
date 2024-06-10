import os
import torch
import json
import numpy as np
from collections import defaultdict


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        print(f"Loading checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None
    

def save_model(model, filename):
    print(f"Saving model state dict to {filename}")
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    print(f"Loading model state dict from {filename}")
    model.load_state_dict(torch.load(filename))
    return model


def save_model_checkpoint(model, optimizer, epoch, filename):
    print(f"Saving model and optimizer state dicts to {filename}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

def load_model_checkpoint(model, optimizer, filename):
    print(f"Loading model and optimizer state dicts from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

def save_model_checkpoint_wandb(model, optimizer, epoch, filename, wandb):
    print(f"Saving model and optimizer state dicts to {filename}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    wandb.save(filename)

def load_model_checkpoint_wandb(model, optimizer, filename, wandb):
    print(f"Loading model and optimizer state dicts from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

def save_model_wandb(model, filename, wandb):
    print(f"Saving model state dict to {filename}")
    torch.save(model.state_dict(), filename)
    wandb.save(filename)

def load_model_wandb(model, filename, wandb):
    print(f"Loading model state dict from {filename}")
    model.load_state_dict(torch.load(filename))
    return model


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data



def generate_syn_logistic(dimension, n_clnt, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):

    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_clnt)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))

    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_clnt))
    data_y = list(range(n_clnt))

    mean_W = np.random.normal(0, alpha, n_clnt)
    B = np.random.normal(0, beta, n_clnt)

    mean_x = np.zeros((n_clnt, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_clnt):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))

    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))

    for i in range(n_clnt):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y


__all__ = ['save_checkpoint', 'load_checkpoint', 'save_model', 
           'load_model', 'save_model_checkpoint', 'load_model_checkpoint',
             'save_model_checkpoint_wandb', 'load_model_checkpoint_wandb', 
             'save_model_wandb', 'load_model_wandb', 'read_dir', 'read_data', 
             'generate_syn_logistic']