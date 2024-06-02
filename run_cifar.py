import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
from tqdm import tqdm
import wandb

import argparse




def args_parser():
    parser = argparse.ArgumentParser()
    # location argument
    parser.add_argument('--shakespeare_train_path', type=str, default='', help="training path for shakespeare dataset")
    parser.add_argument('--shakespeare_test_path', type=str, default='', help="test path for shakespeare dataset")
    parser.add_argument('--cifar_dataset', type=str, default='./data/cifar', help="path for cifar dataset")

    # federated arguments
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=4, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=4*10^-3, help="learning rate decay each round")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='lstm', help='model name')
    parser.add_argument('--input_size', type=int, default=80, help='model input size')
    parser.add_argument('--embedding_size', type=int, default=8, help='model embedding size')
    parser.add_argument('--hidden_size', type=int, default=256, help='model hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='model number of layers')
    parser.add_argument('--output_size', type=int, default=80, help='model output size')

    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--wandb_key', type=str, default='', help='wandb key')

    args = parser.parse_args()
    return args
    



args= args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


if args.wandb_key:
  wandb.login(key=args.wandb_key)

    # Initialize wandb
  wandb.init(project='fed-2', entity='developer-sidani')
else:
    print("wandb key not provided, logging is disabled")



import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


if args.dataset == 'cifar':
    #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_cifar_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_cifar_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR100(args.cifar_dataset, train=True, download=True, transform=trans_cifar_train)
    dataset_test = datasets.CIFAR100(args.cifar_dataset, train=False, download=True, transform=trans_cifar_test)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users = cifar_noniid(dataset_train, args.num_users)

import torch
from torch import nn
import torch.nn.functional as F

class CIFARLeNet(nn.Module):
    def __init__(self):
        super(CIFARLeNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),  # Dropout layer after first Max Pooling
            # Conv Layer 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)  # Dropout layer after second Max Pooling
        )

        self.fc_layers = nn.Sequential(
            # FC Layer 1
            nn.Linear(64*8*8, 384),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer after first FC layer
            # FC Layer 2
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer after second FC layer
            # Output Layer
            nn.Linear(192, 100)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

net_glob = CIFARLeNet().to(args.device)
print(net_glob)

#training
net_glob.train()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.lr = args.lr
        self.lr_decay = args.lr_decay

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        epoch_loss = []
        for iter in tqdm(range(self.args.local_ep), desc="Epochs"):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.ldr_train, desc="Batches", leave=False)):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(tqdm(data_loader, desc="Processing")):
      if torch.cuda.is_available() and args.gpu != -1:
          data, target = data.cuda(args.device), target.cuda(args.device)
      else:
          data, target = data.cpu(), target.cpu()
      
      log_probs = net_g(data)
      # Sum up batch loss
      test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
      # Get the index of the max log-probability
      y_pred = log_probs.data.max(1, keepdim=True)[1]
      correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

if args.wandb_key:
    wandb.config.update(args)
    wandb.watch(net_glob)

# copy weights
w_glob = net_glob.state_dict()

# training
acc_test = []
clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            for idx in range(args.num_users)]
m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
for iter in range(args.epochs):
    w_locals, loss_locals, weight_locols= [], [], []
    idxs_users = np.random.choice(clients_index_array, m, replace=False)
    for idx in idxs_users:
        w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        weight_locols.append(len(dict_users[idx]))

    # update global weights
    w_glob = FedWeightAvg(w_locals, weight_locols)
    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    # print accuracy
    net_glob.eval()
    acc_t, loss_t = test_img(net_glob, dataset_test, args)
    if args.wandb_key:
        wandb.log({'Loss': loss_t, 'Round': iter, 'Accuracy': acc_t})
    print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))

    acc_test.append(acc_t.item())


rootpath = './log'
if not os.path.exists(rootpath):
    os.makedirs(rootpath)
accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}.dat'.
                format(args.dataset, args.model, args.epochs, args.iid), "w")

for ac in acc_test:
    sac = str(ac)
    accfile.write(sac)
    accfile.write('\n')
accfile.close()