import torch
from torch import nn
from torch.utils.data import DataLoader


class LocalUpdate(object):
    def __init__(self, args, client_train, val_set):
        self.args = args
        self.trainloader, self.valloader = self.data_loaders(client_train, val_set)
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    #def train_val_test(self, dataset, idxs):
    def data_loaders(self, client_train, val_set):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """

        trainloader = DataLoader(client_train,
                                 batch_size=self.args.local_bs,  # use 64
                                 shuffle=True)
        valloader = DataLoader(val_set,
                                 batch_size=self.args.local_bs,  # use 64
                                 shuffle=False)

        return trainloader, valloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        round_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=4e-4)

        rounds_counter = 0
        rounds = self.args.local_ep  # Define the number of rounds
        while rounds_counter < rounds:
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if rounds_counter >= rounds:
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Round : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, rounds_counter, batch_idx * len(images),
                        len(self.trainloader),
                        100. * batch_idx / len(self.trainloader), loss.item()))

                round_loss.append(loss.item())
                rounds_counter += 1

        average_loss = sum(round_loss) / len(round_loss)
        return model.state_dict(), average_loss

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.valloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

__all__ = ["LocalUpdate", "test_inference"]
