#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np

def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data

#
def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    sample_size = batch_size * (len(vocabulary) // batch_size)
    out_array = np.array(vocabulary[:sample_size])
    out_array = out_array.reshape([batch_size, -1])  

    sample_size = out_array.shape[1]
    len_count = random.randint(0,16)

    while True:
      length = num_steps + 1
      if length + len_count > sample_size:
        break

      yield out_array[: ,len_count: len_count+length]
      len_count = len_count + length


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
