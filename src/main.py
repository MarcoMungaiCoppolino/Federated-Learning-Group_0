from utils.options import *
# from utils.sampling import *
# from utils.get_functions import *

def main_runner():
    args = args_parser()
            
    # Get user input for parameters
    iid, participation, Nc, J = get_user_input()
    
    # Override the args with user input
    args.iid = iid
    args.participation = participation
    args.Nc = Nc
    args.local_ep = J

    exp_details(args)  
        
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    
    train_set, test_set, user_groups_train, user_groups_val, user_groups_test = get_dataset(args)
    
    global_model = LeNet5(args=args)
    
    
    global_model.to(device)
    global_model.train()
    print(global_model)
    
    # copy weights
    global_weights = global_model.state_dict()
    
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
    
        global_model.train()
        # args.frac = C = 0.1
        num_selected_clients = max(int(args.frac * args.num_users), 1)
    
        if args.participation:
            # Uniform participation
            idxs_users = np.random.choice(range(args.num_users), num_selected_clients, replace=False)
        else:
            # Skewed participation
            client_probabilities = np.random.dirichlet([args.gamma] * args.num_users)
            idxs_users = np.random.choice(range(args.num_users), size=num_selected_clients, p=client_probabilities, replace=False)
    
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, client_train=user_groups_train[idx],
                                      client_val=user_groups_val[idx], 
                                      client_test=user_groups_test[idx])
    
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
    
        # update global weights
        global_weights = average_weights(local_weights)
    
        # update global weights
        global_model.load_state_dict(global_weights)
    
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, client_train=user_groups_train[idx],
                                      client_val=user_groups_val[idx], 
                                      client_test=user_groups_test[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
    
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
    
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_set)
    
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))


if __name__ == '__main__':
    main_runner()
