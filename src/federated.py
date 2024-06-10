from utils import *
from utils.options import *
from utils.data_utils import *
from utils.update import *
from utils.sampling import *
from utils.exp_details import *
from utils.average_weights import *
import os
import copy
import numpy as np
from tqdm import tqdm
from models import *
import glob 
import torch
from utils.wandb_utils import *


if __name__ == '__main__':
    args = args_parser()
    wandb_logger = WandbLogger(args)

    if args.gpu:
        d = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        print(f"Using {d} device")
        torch.cuda.set_device(d)
    device = 'cuda' if args.gpu else 'cpu'
    
    train_set, val_set, test_set, user_groups_train = get_dataset(args)
    
    if args.dataset == 'cifar':

        if args.iid:
            if args.partecipation:
                checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_epoch_*.pth.tar"
            else:
                checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.gamma}_epoch_*.pth.tar"

        else:
            if args.partecipation:
                checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"

            else:
                checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.gamma}_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"
    else:
        checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"
    # Find the latest checkpoint that matches the pattern
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = load_checkpoint(latest_checkpoint)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            last_train_accuracy = checkpoint['train_accuracy']
            last_user_input = checkpoint['user_input']
            train_loss = checkpoint['train_loss']

            # Print the status of the last checkpoint
            participation_status = 'uniform' if last_user_input[1] == 1 else 'skewed'
            if last_user_input[0] == 0:  # Check if iid parameter is 0
                user_input_string = f"IID: {last_user_input[0]}, Participation: {participation_status}, Nc: {last_user_input[2]}, J: {last_user_input[3]}"
            else:
                user_input_string = f"IID: {last_user_input[0]}, Participation: {participation_status}, J: {last_user_input[3]}"

            print(f"\nA saving checkpoint with these parameters exists:\n"
                f"Last checkpoint details:\n"
                f"Epoch reached: {start_epoch}\n"
                f"Train accuracy: {100*last_train_accuracy[-1]}%\n"
                f"Training loss: {np.mean(np.array(train_loss))}\n"
                f"User input variables: {user_input_string}\n")

            # Ask the user if they want to continue from the last checkpoint or start again
            

            if args.checkpoint_resume == 1:
                args.iid, args.partecipation, args.Nc, args.local_ep = last_user_input
                global_model = CIFARLeNet() if args.dataset == 'cifar' else ShakespeareLSTM(args=args)
                global_model.to(device)
                global_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Resuming training from epoch {start_epoch}")
            else:
                start_epoch = 0
                global_model = CIFARLeNet() if args.dataset == 'cifar' else ShakespeareLSTM(args=args)
                global_model.to(device)
        else:
            start_epoch = 0
            global_model = CIFARLeNet() if args.dataset == 'cifar' else ShakespeareLSTM(args=args)
            global_model.to(device)
    else:
        start_epoch = 0
        global_model = CIFARLeNet() if args.dataset == 'cifar' else ShakespeareLSTM(args=args)
        global_model.to(device)
    
    global_model.train()
    print(global_model, "\n")

    
    wandb_logger.watch(global_model)
    # Copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    print_every = args.print_every

    # Initialize tqdm with the starting epoch
    with tqdm(total=args.epochs, initial=start_epoch, desc="Training") as pbar:
        for epoch in range(start_epoch, args.epochs):
            local_weights, local_losses = [], []
            print(f'\n\n| Global Training Round : {epoch+1} |')

            global_model.train()
            num_selected_clients = max(int(args.frac * args.num_users), 1)

            if args.partecipation:
                # Uniform participation
                idxs_users = np.random.choice(range(args.num_users), num_selected_clients, replace=False)
            else:
                # Skewed participation
                client_probabilities = np.random.dirichlet([args.gamma] * args.num_users)
                idxs_users = np.random.choice(range(args.num_users), size=num_selected_clients, p=client_probabilities, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(args=args, client_train=user_groups_train[idx], val_set=val_set)

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # Update global weights
            global_weights = average_weights(local_weights)

            # Update global model
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            global_model.eval()

            if (epoch+1) % print_every == 0:
                local_model = LocalUpdate(args=args, client_train=user_groups_train[idx], val_set=val_set)
                acc, loss = local_model.inference(model=global_model)
                train_accuracy.append(acc)
                # Print global training loss after every 'print_every' rounds
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(train_loss))}')
                wandb_logger.log({'Loss': np.mean(np.array(train_loss)), 'Round': epoch+1, 'Accuracy': 100*train_accuracy[-1]})
                print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

            if (epoch+1) % print_every == 0:
                # Save checkpoint
                if args.iid:
                    filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.local_ep}_epoch_{epoch+1}.pth.tar"
                else:
                    filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.Nc}_{args.local_ep}_epoch_{epoch+1}.pth.tar"

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': global_model.state_dict(),
                    'loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'user_input': (args.iid, args.partecipation, args.Nc, args.local_ep),
                    'train_loss': train_loss
                }
                save_checkpoint(checkpoint, filename=filename)

                # Remove the previous checkpoint unless it's a multiple of the backup parameter
                if (epoch + 1) > print_every:
                    if (epoch + 1 -10) % args.backup != 0:
                        prev_epoch = epoch + 1 - print_every

                        if args.iid:
                            prev_filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                        else:
                            prev_filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.partecipation}_{args.Nc}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                        if os.path.exists(prev_filename):
                            os.remove(prev_filename)

            # Update the progress bar
            pbar.update(1)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_set)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
