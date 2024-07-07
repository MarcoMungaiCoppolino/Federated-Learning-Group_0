import torch.nn as nn
import torch.nn.functional as F

class CIFARLeNet(nn.Module):
    """
    A neural network model inspired by LeNet5, designed for CIFAR-100 dataset.

    Attributes:
    ----------
    flatten : nn.Module
        A layer to flatten the input tensor.
    conv1 : nn.Module
        First convolutional layer with 3 input channels and 64 output channels.
    conv2 : nn.Module
        Second convolutional layer with 64 input channels and 64 output channels.
    pool : nn.Module
        Max pooling layer with kernel size of 2.
    fc1 : nn.Module
        Fully connected layer with input size 64*5*5 and output size 384.
    fc2 : nn.Module
        Fully connected layer with input size 384 and output size 192.
    fc3 : nn.Module
        Fully connected layer with input size 192 and output size 100.
    """

    def __init__(self):
        """
        Initialize the CIFARLeNet model with its layers.
        """
        super(CIFARLeNet, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100)
    def produce_feature(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor.

        Returns:
        -------
        torch.Tensor
            The output tensor after applying all the layers.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden