import os
import torch

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

__all__ = ['save_checkpoint', 'load_checkpoint', 'save_model', 'load_model', 'save_model_checkpoint', 'load_model_checkpoint', 'save_model_checkpoint_wandb', 'load_model_checkpoint_wandb', 'save_model_wandb', 'load_model_wandb']