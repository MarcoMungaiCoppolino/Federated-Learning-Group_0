import re
import json
import numpy as np
import os
from collections import defaultdict


ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def batch_data(data, batch_size, seed):
    '''
    Shuffles and divides the data into batches.

    Parameters:
    data (dict): A dictionary containing 'x' and 'y' keys with numpy array values.
    batch_size (int): The size of each batch.
    seed (int): Random seed for shuffling.

    Yields:
    tuple: Pairs of numpy arrays (batched_x, batched_y) each of length batch_size.
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)
        # 'yield' method returns one pair at a time and doesn't stop the function, it waits for the next call.

def process_x(raw_x_batch):
    '''
    Converts a batch of words to indices.

    Parameters:
    raw_x_batch (list): A list of words.

    Returns:
    numpy array: A numpy array of lists where each list contains indices of the letters in the corresponding word.
    '''
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    '''
    Converts a batch of letters to one-hot encoded vectors.

    Parameters:
    raw_y_batch (list): A list of letters.

    Returns:
    list: A list of one-hot encoded vectors corresponding to the letters.
    '''
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch

def _one_hot(index, size):
    '''
    Creates a one-hot encoded vector.

    Parameters:
    index (int): The index to be set to 1.
    size (int): The size of the one-hot vector.

    Returns:
    list: A one-hot encoded vector of given size with a 1 at the specified index.
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec

def letter_to_vec(letter):
    '''
    Converts a letter to a one-hot encoded vector.

    Parameters:
    letter (str): A single character.

    Returns:
    list: A one-hot encoded vector representing the letter.
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)

def word_to_indices(word):
    '''
    Converts a word to a list of indices.

    Parameters:
    word (str): A word.

    Returns:
    list: A list of indices representing the positions of the letters in the word.
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

def read_dir(data_dir):
    '''
    Reads data from a directory of JSON files.

    Parameters:
    data_dir (str): The directory containing JSON files.

    Returns:
    tuple: A tuple containing lists of clients, groups, and a dictionary of data.
    '''
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    '''
    Reads training and test data from directories of JSON files.

    Parameters:
    train_data_dir (str): The directory containing training data JSON files.
    test_data_dir (str): The directory containing test data JSON files.

    Returns:
    tuple: A tuple containing lists of clients, groups, and dictionaries of training and test data.
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


__all__ = ['batch_data', 'process_x', 'process_y', 'read_data', 'read_dir', 'letter_to_vec', 'word_to_indices']