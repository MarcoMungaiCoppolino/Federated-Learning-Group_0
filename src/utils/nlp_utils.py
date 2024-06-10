import numpy as np
import json
import numpy as np
import re


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)

def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
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

__all__ = ['split_line', 'line_to_indices', 'bag_of_words', 'get_word_emb_arr', 'val_to_vec', 'batch_data']