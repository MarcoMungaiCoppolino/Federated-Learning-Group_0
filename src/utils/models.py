import torch
import os
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import functional as F

def eval_pfedhn(nodes, num_nodes, hnet, net, criterion, device, loader_type):
    
    @torch.no_grad()
    def evaluate(nodes, num_nodes, hnet, net, criterion, device, loader_type):
        hnet.eval()
        results = defaultdict(lambda: defaultdict(list))

        for node_id in range(num_nodes):  # iterating over nodes
            running_loss, running_correct, running_samples = 0., 0., 0.
            if loader_type == 'test':
                curr_data = nodes[node_id].test_dataloader
            elif loader_type == 'val':
                curr_data = nodes[node_id].val_dataloader
            else:
                curr_data = nodes[node_id].train_dataloader

            for batch_count, batch in enumerate(curr_data):
                img, label = tuple(t.to(device) for t in batch)

                weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
                net.load_state_dict(weights)
                pred = net(img)
                running_loss += criterion(pred, label).item()
                running_correct += pred.argmax(1).eq(label).sum().item()
                running_samples += len(label)

            results[node_id]['loss'] = running_loss / (batch_count + 1)
            results[node_id]['correct'] = running_correct
            results[node_id]['total'] = running_samples
            results[node_id]['accuracy'] = running_correct / running_samples

        return results

    curr_results = evaluate(nodes, num_nodes, hnet, net, criterion, device, loader_type=loader_type)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = np.mean([val['accuracy'] for val in curr_results.values()])
    acc = total_correct / total_samples

    return curr_results, avg_loss, avg_acc, acc



def load_checkpoint(filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        print(f"Loading checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None


def update_weights(model, weights):
    assert len(weights) == len(list(model.parameters())), "Number of weights must match number of model parameters"
    for param, weight in zip(model.parameters(), weights):
        param.data.copy_(weight)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def shakespeare_inference(model, dataloader, criterion, args):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            
            labels = labels.squeeze(1)          
            if args.device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
            hidden = init_hidden(batch_size=inputs.size(0))
            outputs,_ = model(inputs,hidden)            
            loss = criterion(outputs, labels)            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(dataloader)
    accuracy = correct / total
    return accuracy, test_loss


def init_hidden(num_layers = 2, hidden_size = 256, batch_size=64):
        if torch.cuda.is_available():
            return (torch.zeros(num_layers, batch_size, hidden_size).cuda(),
                    torch.zeros(num_layers, batch_size, hidden_size).cuda())
        return (torch.zeros(num_layers, batch_size, hidden_size),
                torch.zeros(num_layers, batch_size, hidden_size))


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

__all__ = ['update_weights', 'load_checkpoint','inference', 'save_checkpoint', 'eval_pfedhn','shakespeare_inference', 'init_hidden']
