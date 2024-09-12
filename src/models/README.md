# PyTorch Neural Network Models

This repository contains several PyTorch neural network models for different tasks. Specifically, it includes implementations for:

- `CIFARLeNet`: A LeNet5-inspired model tailored for the CIFAR-100 dataset.
- `CNNHyper`: A flexible convolutional neural network that learns the weights of its convolutional and fully connected layers.
- `CharLSTM`: A character-level LSTM model designed for sequence prediction tasks.

## Models

### CIFARLeNet

A convolutional neural network inspired by LeNet5, adapted for the CIFAR-100 dataset.

#### Attributes
- `conv1`: First convolutional layer with 3 input channels and 64 output channels.
- `conv2`: Second convolutional layer with 64 input channels and 64 output channels.
- `pool`: Max pooling layer with a kernel size of 2.
- `fc1`: Fully connected layer with input size 64*5*5 and output size 384.
- `fc2`: Fully connected layer with input size 384 and output size 192.
- `fc3`: Fully connected layer with input size 192 and output size 100 (for CIFAR-100 classes).

#### Methods
- `forward(x: torch.Tensor) -> torch.Tensor`: Defines the forward pass of the model. Takes an input tensor and returns the output tensor after applying all the layers.

### CNNHyper

A flexible convolutional neural network model that learns the weights for convolutional and fully connected layers.

#### Attributes
- `in_channels`: Number of input channels for the convolutional layers (default is 3).
- `out_dim`: Output dimension (number of classes for classification).
- `n_kernels`: Number of kernels used in the convolutional layers.
- `embeddings`: Embedding layer to map node indices to dense vectors.
- `mlp`: Multi-layer perceptron (MLP) that processes the embeddings.

#### Methods
- `forward(idx: torch.Tensor) -> OrderedDict`: Defines the forward pass of the model and returns the learned weights for the layers. Takes input indices for the embeddings and returns a dictionary containing the weights and biases for the convolutional and fully connected layers.

### CharLSTM

A character-level LSTM model for sequence prediction tasks.

#### Attributes
- `hidden_size`: The number of features in the hidden state of the LSTM.
- `num_layers`: The number of recurrent layers in the LSTM.
- `embedding`: Embedding layer that maps character indices to dense vectors.
- `lstm`: The LSTM layer that processes the embedded input.
- `fc`: A fully connected layer that maps LSTM output to the output size.

#### Methods
- `forward(x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Defines the forward pass of the model. Takes an input tensor containing character indices and the hidden state, and returns the output tensor and the updated hidden state.
