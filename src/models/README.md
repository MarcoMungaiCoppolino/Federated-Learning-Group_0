# Overview

This directory contains two PyTorch models: `CIFARLeNet` and `CharLSTM`. These models are implemented using the `torch.nn.Module` base class and are designed for different types of tasks.

## CIFARLeNet

### Description

`CIFARLeNet` is a convolutional neural network model inspired by LeNet. It is specifically designed for image classification tasks using the CIFAR-100 dataset.

### Architecture

- **Layers**:
  - `flatten`: Flattens the input tensor.
  - `conv1`: First convolutional layer with 3 input channels and 64 output channels, using a kernel size of 5.
  - `conv2`: Second convolutional layer with 64 input channels and 64 output channels, using a kernel size of 5.
  - `pool`: Max pooling layer with a kernel size of 2.
  - `fc1`: Fully connected layer with input size of 64 * 5 * 5 and output size of 384.
  - `fc2`: Fully connected layer with input size of 384 and output size of 192.
  - `fc3`: Fully connected layer with input size of 192 and output size of 100.

### Usage

To use the `CIFARLeNet` model for image classification tasks on CIFAR-100 dataset, follow these steps:

1. Initialize an instance of `CIFARLeNet`.
2. Prepare the data in the appropriate format (e.g., normalize and batchify).
3. Forward pass the input through the model to get predictions.

```python
from cifar_lenet import CIFARLeNet

# Example usage
model = CIFARLeNet()
# Prepare your CIFAR-100 dataset and DataLoader
# Forward pass example
outputs = model(inputs)
```

## CharLSTM

### Description

`CharLSTM` is a character-level LSTM model designed for sequence prediction tasks. It can be used for tasks such as text generation or character-level language modeling.

### Architecture

- **Layers**:
  - `embedding`: Embedding layer to convert input indices to dense vectors.
  - `lstm`: LSTM layer for sequence processing, with specified embedding size, hidden size, and number of layers.
  - `fc`: Fully connected layer to produce the final output.

### Usage

To use the `CharLSTM` model for sequence prediction tasks, follow these steps:

1. Initialize an instance of `CharLSTM` with appropriate parameters.
2. Prepare the input sequences and initial hidden state.
3. Forward pass the input through the model to get predictions and updated hidden state.

```python
from char_lstm import CharLSTM

# Example usage
model = CharLSTM(input_size, embedding_size, hidden_size, num_layers, output_size)
# Prepare your input sequences and initial hidden state
# Forward pass example
outputs, new_hidden = model(inputs, initial_hidden)
```

---

This `README.md` provides a high-level overview of each model's purpose, architecture, and usage instructions. Replace `cifar_lenet` and `char_lstm` with the appropriate filenames where your model implementations reside. Adjust the usage examples to fit your specific use case and dataset handling procedures.