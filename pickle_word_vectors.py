"""
This script loads word embeddings in a txt format and saves them as a pickled python dictionary for a faster loading.
The loading of the whole GloVe embeddings from a pickle format can take as little as few seconds!
"""

import os
import sys
import pickle
import functools

import gensim
import numpy as np


def load_bio_aqs_format(directory):
    tokens_filename = os.path.join(directory, 'types.txt')
    vectors_filename = os.path.join(directory, 'vectors.txt')

    word_vectors = {}
    with open(tokens_filename, 'r') as f_t, open(vectors_filename, 'r') as f_v:
        for token, vector_str in zip(f_t, f_v):
            token = token.strip()
            vector_str = vector_str.strip()

            vector = [float(ve) for ve in vector_str.split()]
            vector = np.array(vector, dtype=np.float32)

            word_vectors[token] = vector

    return word_vectors


def load_glove_format(filename):
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()

            try:
                word = line[0]
                word_vector = np.array([float(v) for v in line[1:]])
            except ValueError:
                continue

            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)

            if len(word_vector) != embeddings_dim:
                continue

            word_vectors[word] = word_vector

    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())

    return word_vectors


LOAD_FUNCTIONS = {
    'word2vec_txt': gensim.models.KeyedVectors.load_word2vec_format,
    'model': gensim.models.word2vec.Word2Vec.load,
    'bio_asq': load_bio_aqs_format,
    'glove': load_glove_format,
}


def load_word_vectors(filename, target_type=None):
    if target_type is None:
        _, extension = os.path.splitext(filename)

        if extension == '.txt':
            target_type = 'word2vec_txt'
        elif extension == '.vec':
            target_type = 'word2vec_txt'
        elif extension == '.model':
            target_type = 'model'

    if target_type is None:
        raise ValueError('Cannot determine the type of the input files')

    load_fn = LOAD_FUNCTIONS[target_type]
    try:
        word_vectors = load_fn(filename)
    except ValueError as e:
        print('Error:', e)
        word_vectors = {}

    if not isinstance(word_vectors, dict):
        if hasattr(word_vectors, 'index2word'):
            word_vectors = {w: word_vectors[w] for w in word_vectors.index2word}
        else:
            raise ValueError('Cannot convert word vectors to dict')

    print('Loaded:', len(word_vectors))
    return word_vectors


def main(input_filename, output_filename, nb_tokens=None, target_type=None):
    word_vectors = load_word_vectors(input_filename, target_type=target_type)

    if nb_tokens is not None:
        word_vectors = {w: word_vectors[w] for w in list(word_vectors.keys())[:nb_tokens]}

    with open(output_filename, 'wb') as f:
        pickle.dump(word_vectors, f)

    print('Saved:', os.path.basename(output_filename))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        help_string = 'Convert a word vevctor .txt/gensim file into a pickle file' \
                      '\n\n' \
                      'Usage: {} <input_filename> <output_filename>'.format(os.path.basename(__file__))

        print(help_string)
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # save only top `nb_tokens` tokens
    nb_tokens = None
    if len(sys.argv) >= 4:
        nb_tokens = int(sys.argv[3])
        if nb_tokens == -1:
            nb_tokens = None

    target_type = None
    if len(sys.argv) >= 5:
        target_type = sys.argv[4]

    main(input_filename, output_filename, nb_tokens=nb_tokens, target_type=target_type)
