import torch
import os
from torch.utils.data import DataLoader



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
            inputs, labels = inputs.cuda(), labels.cuda() if args.device == 'cuda' else  inputs, labels # Move data to CUDA
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = test_loss / len(testloader)
    accuracy = correct / total
    return accuracy, test_loss

__all__ = ['initialize_hidden_state', 'update_weights', 'load_checkpoint','inference', 'save_checkpoint']