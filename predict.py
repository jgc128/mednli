import os
import sys
import logging
import tempfile
import select

import numpy as np

from sacred import Experiment

from config import DATA_DIR, MODELS_WEIGHTS_DIR
from utils.io import load_pickle, save_pickle, get_word_vectors_filename
from train_model import create_model
from preprocess import create_data_matrices

ex = Experiment('predict')


def load_model(model_class, model_weights_filename, embeddings_filename, max_len=50):
    model_params = {
        'max_len': max_len,
        'rnn_size': 300,
        'hidden_size': 300,
        'rnn_cell': 'LSTM',
        'regularization': 0.000001,
        'dropout': 0.5,
        'trainable_embeddings': False,
    }

    # load embeddings
    W_emb = load_pickle(embeddings_filename)
    model_params['W_emb'] = W_emb
    logging.info('Embeddings restored: %s', os.path.basename(embeddings_filename))

    # create model
    model = create_model(model_class, model_params)

    # load weights
    model.load_weights(model_weights_filename)
    logging.info('Weights restored: %s', os.path.basename(model_weights_filename))

    return model


def check_stdin():
    """https://stackoverflow.com/a/3763257"""
    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        return True
    else:
        return False


def get_input_sentences():
    if not check_stdin():
        logging.error('Provide input sentences in STDIN as \\t-delimited strings')
        sys.exit(1)

    inputs = []
    for line in sys.stdin:
        sentences = line.strip().split('\t')
        if len(sentences) == 2:
            inputs.append(sentences)

    return inputs


def get_prediction(model, tokenizer, premise, hypothesis, max_len=50):
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    # convert the data
    data = {
        'premise': premise,
        'hypothesis': hypothesis,
        'label': [1, ] * len(premise),
    }
    m_input = create_data_matrices(tokenizer, data, max_len=max_len, padding='post')

    # predict
    predictions = model.predict(m_input)
    probabilities = softmax(predictions)

    probabilities = probabilities.tolist()

    return probabilities


@ex.config
def config():
    model_class = 'PyTorchInferSentModel'
    model_weights_filename = 'PyTorchInferSentModel_50_glove_bio_asq_mimic_clinical__.slysamwq.h5'
    tokenizer_filename = 'tokenizer_clinical_.pickled'
    embeddings_filename = 'embeddings_clinical_.pickled'


@ex.main
def main(model_class, model_weights_filename, tokenizer_filename, embeddings_filename, _config):
    model_weights_filename = MODELS_WEIGHTS_DIR.joinpath(model_weights_filename)
    tokenizer_filename = DATA_DIR.joinpath(tokenizer_filename)
    embeddings_filename = DATA_DIR.joinpath(embeddings_filename)

    model = load_model(model_class, model_weights_filename, embeddings_filename)
    tokenizer = load_pickle(tokenizer_filename)

    input_sentences = get_input_sentences()
    premise, hypothesis = zip(*input_sentences)

    probabilities = get_prediction(model, tokenizer, premise, hypothesis)
    np.savetxt(sys.stdout, probabilities)


if __name__ == '__main__':
    ex.run_commandline()
