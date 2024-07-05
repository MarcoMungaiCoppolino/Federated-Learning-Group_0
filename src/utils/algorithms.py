import glob 
from utils.models import *
import numpy as np
import torch
import torch.optim as optim
from models import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def fedAVG(global_model, user_groups_train, criterion, args, logger, metrics, wandb_logger, device, test_set):
    clients_distribs = {client.client_id: 0 for client in user_groups_train}
    if args.iid:
        if args.participation:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_epoch_*.pth.tar"
        else:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.gamma}_epoch_*.pth.tar"

    else:
        if args.participation:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"

        else:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"
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

            logger.info(f"\nA saving checkpoint with these parameters exists:\n"
                f"Last checkpoint details:\n"
                f"Epoch reached: {start_epoch}\n"
                f"Train accuracy: {100*last_train_accuracy[-1]}%\n"
                f"Training loss: {np.mean(np.array(train_loss))}\n"
                f"User input variables: {user_input_string}\n")

            # Ask the user if they want to continue from the last checkpoint or start again
            

            if args.checkpoint_resume == 1:
                # args.iid, args.participation, args.Nc, args.local_ep = last_user_input
                global_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Resuming training from epoch {start_epoch}")
            else:
                start_epoch = 0
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    wandb_logger.watch(global_model)
    dirichlet_probs = np.random.dirichlet([args.gamma] * len(user_groups_train))
    for client in user_groups_train:
        logger.info(f'Client {client.client_id} has a probability of {dirichlet_probs[client.client_id]}')

    with tqdm(total=args.epochs, initial=start_epoch, desc="Training") as pbar:
        for epoch in range(start_epoch, args.epochs):
            logger.info(f'\n\n| Global Training Round : {epoch+1} |')
            global_weights = [param.clone().detach() for param in global_model.parameters()]

            if args.participation:
                idx_users  = np.random.choice(user_groups_train, int(len(user_groups_train) * args.frac), p=None)
            else:
                idx_users = np.random.choice(user_groups_train, int(len(user_groups_train) * args.frac), p=dirichlet_probs)

            for user in idx_users:
                clients_distribs[user.client_id] = 1
            user_weights = []
            for idx in idx_users:
                if args.dataset == 'cifar':
                    local_model = CIFARLeNet().to(device)
                else:
                    local_model = CharLSTM().to(device)
                update_weights(local_model, global_weights)
                optimizer = optim.SGD(local_model.parameters(), lr=args.lr, weight_decay=4e-3)
                local_model = idx.train(local_model, criterion, optimizer, args)
                user_weights.append([param.clone().detach() for param in local_model.parameters()])

            aggregated_weights = []
            for weights_list in zip(*user_weights):
                aggregated_weight = torch.mean(torch.stack(weights_list), dim=0)
                aggregated_weights.append(aggregated_weight)
            update_weights(global_model, aggregated_weights)
            
            if (epoch+1) % args.print_every == 0:
                acc, loss = inference(global_model, test_set, criterion,args)
                metrics.loc[len(metrics)] = [epoch+1, acc, loss]
                logger.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
                logger.info(f'Test Loss: {loss} Test Accuracy: {100*acc}%')
                wandb_logger.log({
                        'Test Loss': loss,
                        'Test Accuracy': acc * 100,
                        'Round': epoch + 1
                    })
            if (epoch+1) % args.print_every == 0:
                    # Save checkpoint
                    if args.iid:
                        filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.local_ep}_epoch_{epoch+1}.pth.tar"
                    else:
                        filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_{epoch+1}.pth.tar"

                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': global_model.state_dict(),
                        'loss': loss,
                        'user_input': (args.iid, args.participation, args.Nc, args.local_ep),
                        'accuracy': acc,
                    }
                    save_checkpoint(checkpoint, filename=filename)

                    # Remove the previous checkpoint unless it's a multiple of the backup parameter
                    if (epoch + 1) > args.print_every:
                        if (epoch + 1 -10) % args.backup != 0:
                            prev_epoch = epoch + 1 - args.print_every

                            if args.iid:
                                prev_filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                            else:
                                prev_filename = f"{args.checkpoint_path}/checkpoint_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                            if os.path.exists(prev_filename):
                                os.remove(prev_filename)
    # Plot the frequency of client selection
    plt.figure(figsize=(10, 6))

    # Normalize the selection counts
    normalized_counts = [count / sum(clients_distribs.values()) for count in clients_distribs.values()]

    # Create the bar plot
    plt.bar(clients_distribs.keys(), normalized_counts)
    plt.xlabel('Client ID')
    plt.ylabel('Relative frequency')
    if args.participation:  
        plt.title(f'Clients distribution (random selection)')

    else:
        plt.title(f'Clients distribution (gamma={args.gamma})')
        
    # Save the plot as a PDF file
    if args.iid:
        if args.participation:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.iid}_{args.participation}.pdf'
        else:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.iid}_{args.participation}_{args.gamma}.pdf'
    else:
        if args.participation:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pdf'
        else:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pdf'
    plt.savefig(plot_location)

    # Optionally, clear the figure to free up memory
    plt.clf()

    pbar.update(1)
    if args.iid:
        if args.participation:
            pickle_file = f"{args.metrics_dir}/metrics_{args.iid}_{args.participation}.pkl"
        else:
            pickle_file = f"{args.metrics_dir}/metrics_{args.iid}_{args.participation}_{args.gamma}.pkl"
    else:
        if args.participation:
            pickle_file = f"{args.metrics_dir}/metrics_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pkl"
        else:
            pickle_file = f"{args.metrics_dir}/metrics_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pkl"

    metrics.to_pickle(pickle_file)
    logger.info(f"Metrics saved at {pickle_file}")
    logger.info(f"Plots saved at {plot_location}")
    logger.info("Training Done!")

