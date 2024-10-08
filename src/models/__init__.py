import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from collections import OrderedDict

class CIFARLeNet(nn.Module):
    """
    A convolutional neural network model inspired by LeNet5, specifically designed for the CIFAR-100 dataset.

    Attributes
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
        Fully connected layer with input size 192 and output size 100 (for CIFAR-100 classes).

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Defines the forward pass of the model.
    """

    def __init__(self):
        """
        Initialize the CIFARLeNet model with its layers.
        """
        super(CIFARLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 3, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, 100) after applying all the layers.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class CNNHyper(nn.Module):
    """
    A flexible convolutional neural network model that uses learned weights for convolutional layers.
    
    Attributes
    ----------
    in_channels : int
        Number of input channels for the convolutional layers (default is 3).
    out_dim : int
        Output dimension (number of classes for classification).
    n_kernels : int
        Number of kernels used in the convolutional layers.
    embeddings : nn.Embedding
        Embedding layer to map node indices to dense vectors.
    mlp : nn.Sequential
        Multi-layer perceptron that processes the embeddings.

    Methods
    -------
    forward(idx: torch.Tensor) -> OrderedDict
        Defines the forward pass of the model and returns the weights for the convolutional and fully connected layers.
    """

    def __init__(self, n_nodes, embedding_dim, in_channels=3, out_dim=100, n_kernels=64, hidden_dim=100, spec_norm=False, n_hidden=1):
        """
        Initialize the CNNHyper model with its layers.

        Parameters
        ----------
        n_nodes : int
            The number of nodes in the embedding layer.
        embedding_dim : int
            The dimension of the embeddings.
        in_channels : int
            The number of input channels for the convolutional layers (default is 3).
        out_dim : int
            The output dimension (number of classes for classification).
        n_kernels : int
            The number of kernels used in the convolutional layers.
        hidden_dim : int
            The dimension of the hidden layers in the multi-layer perceptron.
        spec_norm : bool
            Whether to apply spectral normalization to the layers (default is False).
        n_hidden : int
            The number of hidden layers in the multi-layer perceptron (default is 1).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        # Define weight and bias layers for convolutional and fully connected layers
        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 384 * 64 * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 384)
        self.l2_weights = nn.Linear(hidden_dim, 192 * 384)
        self.l2_bias = nn.Linear(hidden_dim, 192)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 192)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        # Apply spectral normalization if specified
        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, idx):
        """
        Defines the forward pass of the model and returns the learned weights for the layers.

        Parameters
        ----------
        idx : torch.Tensor
            The input indices for the embeddings.

        Returns
        -------
        OrderedDict
            A dictionary containing the weights and biases for the convolutional and fully connected layers.
        """
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(384, self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(192, 384),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 192),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights

class CharLSTM(nn.Module):
    """
    A character-level LSTM model for sequence prediction tasks.

    Attributes
    ----------
    hidden_size : int
        The number of features in the hidden state of the LSTM.
    num_layers : int
        The number of recurrent layers in the LSTM.
    embedding : nn.Embedding
        An embedding layer that maps character indices to dense vectors.
    lstm : nn.LSTM
        The LSTM layer that processes the embedded input.
    fc : nn.Linear
        A fully connected layer that maps LSTM output to the output size.

    Methods
    -------
    forward(x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Defines the forward pass of the model.
    """

    def __init__(self):
        """
        Initialize the CharLSTM model with its layers.
        """
        super(CharLSTM, self).__init__()
        input_size = 80  # Number of unique characters
        embedding_size = 8  # Size of the embedding vectors
        hidden_size = 256  # Size of the LSTM hidden state
        num_layers = 2  # Number of LSTM layers
        output_size = 80  # Output size (number of classes, e.g., characters)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, sequence_length) containing character indices.
        hidden : Tuple[torch.Tensor, torch.Tensor]
            The hidden state and cell state from the LSTM.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The output tensor of shape (batch_size, output_size) and the updated hidden state.
        """
        embedded = self.embedding(x)  # Convert indices to embeddings
        output, hidden = self.lstm(embedded, hidden)  # Pass through LSTM
        output = self.fc(output[:, -1, :])  # Get the last output and pass through fully connected layer
        return output, hidden
