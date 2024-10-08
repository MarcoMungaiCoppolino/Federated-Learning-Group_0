import glob 
from utils.models import *
import numpy as np
import torch
import torch.optim as optim
from models import *
from tqdm import tqdm, trange
from collections import defaultdict
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def fedAVG(global_model, clients, criterion, args, logger, metrics, wandb_logger, device, test_set):
    # todo: calculate the validation accuracy and loss for each client and for the global model, update metrics
    test_dataloader = DataLoader(test_set, batch_size=args.local_bs, shuffle=False)
     
    clients_distribs = {client.client_id: 0 for client in clients}
    if args.iid:
        if args.participation:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_epoch_*.pth.tar"
        else:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_epoch_*.pth.tar"
    else:
        if args.participation:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"
        else:
            checkpoint_pattern = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}_epoch_*.pth.tar"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))
    if args.resume_checkpoint:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = load_checkpoint(latest_checkpoint)
        # {
        #         'epoch': step + 1,
        #         'model_state_dict': global_model.state_dict(),
        #         'test_loss': loss,
        #         'user_input': (args.iid, args.participation, args.Nc, args.local_ep),
        #         'test_accuracy': acc,
        #         'test_avg_loss': results['test_avg_loss'][-1],
        #         'test_avg_acc': results['test_avg_acc'][-1],
        #         'val_avg_loss': results['val_avg_loss'][-1],
        #         'val_avg_acc': results['val_avg_acc'][-1],
        #     }
        if checkpoint:
            start_epoch = checkpoint['epoch']
            last_test_accuracy = checkpoint['test_accuracy']
            last_user_input = checkpoint['user_input']
            test_loss = checkpoint['test_loss']

            # Print the status of the last checkpoint
            participation_status = 'uniform' if last_user_input[1] == 1 else 'skewed'
            if last_user_input[0] == 0:  # Check if iid parameter is 0
                user_input_string = f"IID: {last_user_input[0]}, Participation: {participation_status}, Nc: {last_user_input[2]}, J: {last_user_input[3]}"
            else:
                user_input_string = f"IID: {last_user_input[0]}, Participation: {participation_status}, J: {last_user_input[3]}"

            logger.info(f"\nA saving checkpoint with these parameters exists:\n"
                f"Last checkpoint details:\n"
                f"Epoch reached: {start_epoch}\n"
                f"Test accuracy: {100*last_test_accuracy[-1]}%\n"
                f"Test loss: {test_loss}\n"
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
    dirichlet_probs = np.random.dirichlet([args.gamma] * len(clients))

    with tqdm(total=args.epochs, initial=start_epoch, desc="Training") as pbar:
        for epoch in range(start_epoch, args.epochs):
            wandb_logger.log({'epoch': epoch + 1})
            logger.info(f'\n\n| Global Training Round : {epoch+1} |')
            global_weights = [param.clone().detach() for param in global_model.parameters()]

            if args.participation:
                idx_users  = np.random.choice(clients, int(len(clients) * args.frac), p=None)
            else:
                idx_users = np.random.choice(clients, int(len(clients) * args.frac), p=dirichlet_probs)

            for user in idx_users:
                clients_distribs[user.client_id] += 1
            user_weights = []
            for idx in idx_users:
                if args.dataset == 'cifar':
                    local_model = CIFARLeNet().to(device)
                else:
                    local_model = CharLSTM().to(device)
                update_weights(local_model, global_weights)
                optimizer = optim.SGD(local_model.parameters(), lr=args.lr, weight_decay=args.inner_wd)
                local_model = idx.train(local_model, criterion, optimizer, args)
                user_weights.append([param.clone().detach() for param in local_model.parameters()])

            aggregated_weights = []
            for weights_list in zip(*user_weights):
                aggregated_weight = torch.mean(torch.stack(weights_list), dim=0)
                aggregated_weights.append(aggregated_weight)
            update_weights(global_model, aggregated_weights)
            
            loader_type = 'test'
            if (epoch+1) % args.print_every == 0:
                if args.dataset =='cifar':
                    cl_acc_list, cl_loss_list = [], []
                    for cl in clients:
                        cl_acc, cl_loss = cl.inference(global_model, criterion, args)
                        cl_val_acc_list, cl_val_loss_list = [], []
                        cl_val_acc, cl_val_loss = cl.inference(global_model, criterion, args, loader_type='val')
                        cl_val_acc_list.append(cl_val_acc)
                        cl_val_loss_list.append(cl_val_loss)
                        # logger.info(f'Client {cl.client_id} Test Loss: {cl_loss} Test Accuracy: {100*cl_acc}%')
                        cl_acc_list.append(cl_acc)
                        cl_loss_list.append(cl_loss)
                
                acc, loss = inference(global_model, test_set, criterion, args) if args.dataset == 'cifar' else shakespeare_inference(global_model, test_dataloader, criterion, args)
                if args.dataset == 'cifar':
                    metrics.loc[len(metrics)] = [epoch+1, acc, loss, np.mean(cl_acc_list), np.mean(cl_loss_list), np.mean(cl_val_acc_list), np.mean(cl_val_loss_list)]
                # else:
                #     metrics.loc[len(metrics)] = [epoch+1, acc, loss,0,0,0,0]
                logger.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
                logger.info(f'Test Loss: {loss} Test Accuracy: {100*acc}%')
                if args.dataset == 'cifar':
                    logger.info(f'Avg Train Loss: {np.mean(cl_loss_list)} Average Train Accuracy: {np.mean(cl_acc_list)}')
                    logger.info(f'Avg Validation Loss: {np.mean(cl_val_loss_list)} Average Validation Accuracy: {np.mean(cl_val_acc_list)}')
                    wandb_logger.log({
                        'Avg Validation Accuracy': np.mean(cl_val_acc_list) * 100,
                        'Avg Validation Loss': np.mean(cl_val_loss_list),
                    })
                wandb_logger.log({
                        # 'Global Model Train Accuracy': train_acc * 100,
                        # 'Global Model Test Accuracy': test_acc * 100,
                        # 'Test Avg Loss': test_avg_loss,
                        'Test Loss': loss,
                        'Test Accuracy': acc * 100,
                        'step': epoch + 1
                    })
                # Save checkpoint
                if args.iid:
                    if args.participation:
                        filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.local_ep}_epoch_{epoch+1}.pth.tar"
                    else:
                        filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.local_ep}_{args.gamma}_epoch_{epoch+1}.pth.tar"
                else:
                    if args.participation:
                        filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_{epoch+1}.pth.tar"
                    else:
                        filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_{args.gamma}_epoch_{epoch+1}.pth.tar"

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': global_model.state_dict(),
                    'loss': loss,
                    'user_input': (args.iid, args.participation, args.Nc, args.local_ep),
                    'accuracy': acc,
                    'train_accuracy': np.mean(cl_acc_list) * 100 if args.dataset == 'cifar' else acc*100,
                    'train_loss': np.mean(cl_loss_list) if args.dataset == 'cifar' else loss,
                }
                save_checkpoint(checkpoint, filename=filename)

                # Remove the previous checkpoint unless it's a multiple of the backup parameter
                prev_epoch = epoch + 1 - args.print_every
                # Check if we need to delete a previous checkpoint
                if (epoch + 1) > args.print_every and prev_epoch != 1900:
                    # Check if the previous epoch is not a backup epoch
                    if ((epoch + 1 - 10) % args.backup != 0) and ((epoch + 1 - 10) % args.backup != 50) and ((epoch + 1 - 10) % args.backup != 80):
                        # Construct the filename based on whether 'iid' is specified
                        if args.iid:
                            prev_filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                        else:
                            prev_filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                        
                        # Check if the file exists and delete it
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
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.algorithm}_{args.iid}_{args.participation}.pdf'
        else:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}.pdf'
    else:
        if args.participation:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pdf'
        else:
            plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pdf'
    plt.savefig(plot_location)

    # Optionally, clear the figure to free up memory
    plt.clf()

    pbar.update(1)
    if args.iid:
        if args.participation:
            pickle_file = f"{args.metrics_dir}/metrics_{args.algorithm}_{args.iid}_{args.participation}.pkl"
        else:
            pickle_file = f"{args.metrics_dir}/metrics_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}.pkl"
    else:
        if args.participation:
            pickle_file = f"{args.metrics_dir}/metrics_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pkl"
        else:
            pickle_file = f"{args.metrics_dir}/metrics_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pkl"

    metrics.to_pickle(pickle_file)
    logger.info(f"Metrics saved at {pickle_file}")
    logger.info(f"Plots saved at {plot_location}")
    logger.info("Training Done!")

def pFedHN(global_model, clients, criterion, args, logger, metrics, wandb_logger, device, test_set):
    nodes = clients
    clients_distribs = {client.client_id: 0 for client in clients}
    embed_dim = args.embed_dim
    num_nodes = args.n_nodes
    
    if embed_dim == -1:
        embed_dim = int(1 + num_nodes / 4)

    # Ahmad add a check if there is a saving checkpoint if it there is then load the state dict on the global_model and the hypernetwork
    hnet = CNNHyper(num_nodes, embed_dim).to(device)
    net = global_model

    ##################
    # init optimizer #
    ##################
    lr = args.lr
    embed_lr = args.embed_lr
    wd = args.wd

    embed_lr = embed_lr if embed_lr is not None else lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, weight_decay=wd
        )
    }
    optimizer = optimizers[args.optimizer]

    ################
    # init metrics #
    ################
    results = defaultdict(list)

    dirichlet_probs = np.random.dirichlet([args.gamma] * num_nodes)

    step_iter = trange(args.epochs)
    for step in step_iter:
        wandb_logger.log({'epoch': step + 1})
        hnet.train()

        if args.participation:
            # Uniform participation
            node_id = np.random.choice(range(num_nodes))
        else:
            # Skewed participation
            node_id = np.random.choice(range(num_nodes), p=dirichlet_probs)
            
        clients_distribs[node_id] = 1
        
        # produce & load local network weights
        weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
        net.load_state_dict(weights)

        # init inner optimizer
        inner_optim = torch.optim.SGD(
            net.parameters(), lr=args.lr, weight_decay=args.inner_wd
        )

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
        
        # inner updates -> obtaining theta_tilda
        inner_steps = args.local_ep * 10
        for i in range(inner_steps):
            net.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            batch = next(iter(nodes[node_id].train_dataloader))
            img, label = tuple(t.to(device) for t in batch)

            pred = net(img)

            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

            inner_optim.step()

        optimizer.zero_grad()
        
        final_state = net.state_dict()

        # calculating delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        if (step +1) % args.print_every == 0:
            filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_{step+1}.pth.tar"
          
            last_eval = step
            step_results, test_avg_loss, test_avg_acc, test_acc = eval_pfedhn(nodes, num_nodes, hnet, net, criterion, device, loader_type="test")

            logger.info(f"\nStep: {step+1}, AVG Test Loss: {test_avg_loss:.4f},  AVG Test Acc: {test_avg_acc:.4f}, Test Acc: {test_acc:.4f}")
                        
            results['test_avg_loss'].append(test_avg_loss)
            results['test_avg_acc'].append(test_avg_acc)
            results['test_acc'].append(test_acc)

            _, val_avg_loss, val_avg_acc, val_acc  = eval_pfedhn(nodes, num_nodes, hnet, net, criterion, device, loader_type="val")
            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['val_acc'].append(val_acc)

            wandb_logger.log({
                'Step': step + 1,
                'Test Avg Loss': test_avg_loss,
                'Test Avg Acc': test_avg_acc,
                'Test Acc': test_acc,
                'Val Avg Loss': val_avg_loss,
                'Val Avg Acc': val_avg_acc,
                'Val Acc': val_acc
            })
            for key in results:
                logger.info(f"{key}: {results[key][-1]}")
                wandb_logger.log({key: results[key][-1]})
            checkpoint = {
                'epoch': step + 1,
                'model_state_dict': global_model.state_dict(),
                'hn_state_dict': hnet.state_dict(),
                'user_input': (args.iid, args.participation, args.Nc, args.local_ep),
                'test_accuracy': results['test_acc'][-1],
                'test_avg_loss': results['test_avg_loss'][-1],
                'test_avg_acc': results['test_avg_acc'][-1],
                'val_accuracy': results['val_acc'][-1],
                'val_avg_loss': results['val_avg_loss'][-1],
                'val_avg_acc': results['val_avg_acc'][-1]
            }
            save_checkpoint(checkpoint, filename=filename)

            # Remove the previous checkpoint unless it's a multiple of the backup parameter
            prev_epoch = step + 1 - args.print_every
            if (step + 1) > args.print_every and prev_epoch != 1900:
                 if ((step + 1 - 10) % args.backup != 0) and ((step + 1 - 10) % args.backup != 50) and ((step + 1 - 10) % args.backup != 80):
                    prev_filename = f"{args.checkpoint_path}/checkpoint_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}_epoch_{prev_epoch}.pth.tar"
                    
                    if os.path.exists(prev_filename):
                        os.remove(prev_filename)
          
            metrics.loc[len(metrics)] = [step + 1, results['test_acc'][-1], results['test_avg_loss'][-1], results['test_avg_acc'][-1], results['val_acc'][-1], results['val_avg_loss'][-1], results['val_avg_acc'][-1]]
    if step != last_eval:
        _, val_avg_loss, val_avg_acc, val_acc  = eval_pfedhn(nodes, num_nodes, hnet, net, criterion, device, loader_type="val")

        results['test_avg_loss'].append(test_avg_loss)
        results['test_avg_acc'].append(test_avg_acc)
        results['test_acc'].append(test_acc)
        
        step_results, test_avg_loss, test_avg_acc, test_acc = eval_pfedhn(nodes, num_nodes, hnet, net, criterion, device, loader_type="test")
        logger.info(f"\nStep: {step+1}, AVG Test Loss: {test_avg_loss:.4f},  AVG Test Acc: {test_avg_acc:.4f}, Test Acc: {test_acc:.4f}")
        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['val_acc'].append(val_acc)
         
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
    if args.participation:
        pickle_file = f"{args.metrics_dir}/metrics_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pkl"
        plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.algorithm}_{args.iid}_{args.participation}_{args.Nc}_{args.local_ep}.pdf'
    else:
        pickle_file = f"{args.metrics_dir}/metrics_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pkl"
        plot_location = f'{args.metrics_dir}/client_selection_frequency_{args.algorithm}_{args.iid}_{args.participation}_{args.gamma}_{args.Nc}_{args.local_ep}.pdf'
    plt.savefig(plot_location)

    # Optionally, clear the figure to free up memory
    plt.clf()

    metrics.to_pickle(pickle_file)
    logger.info(f"Metrics saved at {pickle_file}")
    logger.info(f"Plots saved at {plot_location}")
    logger.info("Training Done!")
