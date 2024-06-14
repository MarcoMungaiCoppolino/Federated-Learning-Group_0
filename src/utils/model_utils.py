import torch
from torch.utils import data
from data_utils import ShakespeareDataset
import numpy as np
from eval_utls import get_acc_loss


def train_model(args, model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(ShakespeareDataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(args.device)

    for e in range(epoch):

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
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

