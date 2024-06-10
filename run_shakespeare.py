!pip install wandb
!git clone https://github.com/TalwalkarLab/leaf.git
!cd leaf/data/shakespeare && ./preprocess.sh -s iid --iu 0.089 --sf 1.0 -k 2000 -t sample -tf 0.8
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import io
import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import numpy as np
import os
import wandb
import torchvision.models as models




"""Utils for language models."""

import re
import numpy as np
import json



wandb.login(key='13f6c62827c13afef515dd313fe5c67b1c1e1c65')

    # Initialize wandb
wandb.init(project='Federated_Learning', entity='developer-sidani', name='fedavg-shakespear')
# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)
model_output_path = '/content/Output/'
data_dir = '/content/leaf/data/shakespeare/'


class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_sol, iid_data, n_dim, n_clnt, n_cls, avg_data, name_prefix):
        self.dataset = 'synt'
        self.name  = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%s_%s' %(n_dim, n_clnt, n_cls, avg_data,
                alpha, beta, theta, iid_sol, iid_data)

        data_path = 'Data'
        if (not os.path.exists('%s/%s/' %(data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data,
                                        alpha=alpha, beta=beta, theta=theta,
                                        iid_sol=iid_sol, iid_dat=iid_data)
            os.mkdir('%s/%s/' %(data_path, self.name))
            np.save('%s/%s/data_x.npy' %(data_path, self.name), data_x)
            np.save('%s/%s/data_y.npy' %(data_path, self.name), data_y)
        else:
            # Load data
            print('Load')
            data_x = np.load('%s/%s/data_x.npy' %(data_path, self.name), allow_pickle=True)
            data_y = np.load('%s/%s/data_y.npy' %(data_path, self.name), allow_pickle=True)

        for clnt in range(n_clnt):
            print(', '.join(['%.4f' %np.mean(data_y[clnt]==t) for t in range(n_cls)]))

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)

# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure

        self.users = users

        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start+crop_amount]

        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['y'])[start:start+curr_amount]

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
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Change structure to DatasetObject structure

        self.users = users

        tst_data_count_per_clnt = (crop_amount//tst_ratio)
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

        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start+crop_amount]

        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        self.tst_x = list(range(tst_data_count))
        self.tst_y = list(range(tst_data_count))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):

            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
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

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'shakespeare':

            self.X_data = data_x
            self.y_data = data_y

            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()


    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx]
            return x, y



torch.cuda.set_device('cuda:0')


# Global parameters
# os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst

# --- Helper functions

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):

    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

# --- Train functions

def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            wandb.log({"epoch": e+1, "Training Accuracy":acc_trn, "Loss":loss_trn })
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model



### Methods
def train_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0):
    method_name = 'FedAvg'
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if not os.path.exists('/content/Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('/content/Output/%s/%s' %(data_obj.name, method_name))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    if os.path.exists('/content/Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('/content/Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model

            fed_model = model_func()
            fed_model.load_state_dict(torch.load('/content/Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_all[j] = fed_model

        trn_perf_sel = np.load('/content/Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        trn_perf_all = np.load('/content/Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('/content/Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_all = np.load('/content/Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))

        clnt_params_list = np.load('/content/Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))

    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            wandb.log({"Communication sel": i+1, "Test Acc": acc_tst, "Loss": loss_tst})
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            wandb.log({"Communication sel": i+1, "Cent Accuracy": acc_tst, "Loss": loss_tst})
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            tst_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            wandb.log({"Communication all": i+1, "Test Acc": acc_tst, "Loss": loss_tst})
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            wandb.log({"Communication all": i+1, "Cent Accuracy": acc_tst, "Loss": loss_tst})

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '/content/Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1)))
                torch.save(all_model.state_dict(), '/content/Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('/content/Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('/content/Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('/content/Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])

                np.save('/content/Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                np.save('/content/Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('/content/Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('/content/Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('/content/Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('/content/Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('/content/Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('/content/Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all



storage_path = '/content/leaf/data/shakespeare/data/'


name = 'shakepeare'
data_obj = ShakespeareObjectCrop(storage_path, 'shakespeare')


model_name         = 'shakespeare' # Model type
com_amount         = 200
save_period        = 50
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 1
lr_decay_per_round = 1
epoch              = 4
learning_rate      = 0.1
print_per          = 5

# Model function
model_func = lambda : client_model(model_name)
init_model = model_func()
# Initalise the model for all methods or load it from a saved initial model
init_model = model_func()
if not os.path.exists('/content/Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('/content/Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), '/content/Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('/content/Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))

print('FedAvg')

[fed_mdls_sel_FedAvg, trn_perf_sel_FedAvg, tst_perf_sel_FedAvg,
 fed_mdls_all_FedAvg, trn_perf_all_FedAvg,
 tst_perf_all_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round)

plt.figure(figsize=(6, 5))
plt.plot(np.arange(com_amount)+1, tst_perf_all_FedAvg[:,1], label='FedAVG')
plt.ylabel('Test Accuracy', fontsize=16)
plt.xlabel('Communication Rounds', fontsize=16)
plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
plt.grid()
plt.xlim([0, com_amount+1])
plt.title(data_obj.name, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('/content/Output/%s/plot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.show()
