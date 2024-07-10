import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import functional as F



def eval_knnper_grid(client_, weights_grid_, capacities_grid_):
    client_results = np.zeros((len(weights_grid_), len(capacities_grid_)))

    for ii, capacity in enumerate(capacities_grid_):
        client_.capacity = capacity
        client_.clear_datastore()
        client_.build_datastore()
        client_.gather_knn_outputs()

        for jj, weight in enumerate(weights_grid_):
            client_results[jj, ii] = client_.evaluate(weight) * client_.n_test_samples

    return client_results

def compute_embeddings_and_outputs(
        model,
        iterator,
        n_classes,
        apply_softmax=True,
        return_embedding_flag=True,
        embedding_dim=None,
        model_name="cifarlenet",
        device="cuda"
):
    """
    Compute the embeddings and the outputs of all samples in an iterator.

    :param model: PyTorch model to be used for inference.
    :param iterator: DataLoader providing the data samples.
    :param n_classes: Number of output classes.
    :param apply_softmax: If selected, a softmax is applied to the output; otherwise, logits are returned.
    :param return_embedding_flag: If true, embeddings will be computed and returned.
    :param embedding_dim: Dimension of the embeddings.
    :param model_name: Name of the model to select the feature extractor.
    
    :return: Tuple containing embeddings (or None), outputs, and labels.
    """
    model.eval()

    if return_embedding_flag:
        assert embedding_dim is not None, "embedding_dim should be provided when return_embedding_flag is True"

    n_samples = len(iterator.dataset)

    if return_embedding_flag:
        embeddings = np.zeros(shape=(n_samples, embedding_dim), dtype=np.float32)
    else:
        embeddings = None

    outputs = np.zeros(shape=(n_samples, n_classes), dtype=np.float32)
    labels = np.zeros(shape=(n_samples,), dtype=np.uint16)

    with torch.no_grad():
        for x, y, indices in iterator:
            x = x.to(device)
            labels[indices] = y.data.cpu().numpy()

            if return_embedding_flag:
                features = model.produce_feature(x).cpu().numpy()

                embeddings[indices] = features

            outs = model(x)
            if apply_softmax:
                outputs[indices] = F.softmax(outs, dim=1).cpu().numpy()
            else:
                outputs[indices] = outs.cpu().numpy()

    return embeddings, outputs, labels

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
__all__ = ['initialize_hidden_state', 'update_weights', 'load_checkpoint','inference', 'save_checkpoint', 'compute_embeddings_and_outputs', 'eval_knnper_grid']