import sys
import logging
import tempfile

import numpy as np
from keras.preprocessing.text import Tokenizer
import keras.backend as K
import networkx as nx

from sacred import Experiment

from config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_TEMPLATE, MODELS_WEIGHTS_DIR, WORD_VECTORS_FILENAME, \
    PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE, WORD_VECTORS_DIR, UMLS_CONCEPTS_GRAPH_FILENAME, DATA_DIR, \
    RETROFITTING_DIR, PROCESSED_CONCEPTS_DATA_TEST_FILENAME_TEMPLATE
from models import PyTorchSimpleModel, PyTorchESIMModel, PyTorchMultiTargetESIMModel, PyTorchMultiTargetSimpleModel, \
    PyTorchInferSentModel, PyTorchMultiTargetInferSentModel
from models.pytorch_multi_target_base_model import PyTorchMultiTargetBaseModel
from preprocess import create_data_matrices, create_embedding_matrix, create_memory_matrix, create_umls_attention, \
    create_wordnet_attention, create_token_cuis
from utils.data import load_processed_genre_data, load_single_genre_data
from utils.io import load_pickle, save_pickle, get_word_vectors_filename
from utils.text import clean_data, stem_data, replace_cui_data
from utils.file_observer import FileObserver, FileObserverDbOption  # do not delete - imports the class for the -Z flag

ex = Experiment('train_model')


def downsample_data(data, nb_needed):
    nb_premises_needed = nb_needed // 3

    premises_train_all = list(set(data['premise']))
    premises_train_needed = set(np.random.choice(premises_train_all, size=nb_premises_needed, replace=False))
    logging.info('Needed premises: %s', len(premises_train_needed))
    leave_train = [True if p in premises_train_needed else False for p in data['premise']]

    data_downsampled = {}
    data_downsampled['premise'] = [p for i, p in enumerate(data['premise']) if leave_train[i]]
    data_downsampled['hypothesis'] = [p for i, p in enumerate(data['hypothesis']) if leave_train[i]]
    data_downsampled['label'] = [p for i, p in enumerate(data['label']) if leave_train[i]]

    return data_downsampled


def process_data(genre_source, genre_target, genre_tune, max_len, lowercase, stem, clean, downsample_source,
                 word_vectors_type, word_vectors_replace_cui, use_umls_attention, use_token_level_attention,
                 padding='pre'):
    """Load data for the target genres, create and fit tokenizer, and return the input matrices"""

    data_source_train, data_source_dev, data_target_train, data_target_dev, data_tune_train, data_tune_dev = \
        load_processed_genre_data(PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE,
                                  genre_source, genre_target, genre_tune)

    _, _, data_clinical_test = load_single_genre_data(
        PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE, genre='clinical',
        filename_test_template=PROCESSED_CONCEPTS_DATA_TEST_FILENAME_TEMPLATE
    )

    if clean:
        data_source_train = clean_data(data_source_train)
        data_source_dev = clean_data(data_source_dev)

        data_target_train = clean_data(data_target_train)
        data_target_dev = clean_data(data_target_dev)

        data_tune_train = clean_data(data_tune_train)
        data_tune_dev = clean_data(data_tune_dev)

        data_clinical_test = clean_data(data_clinical_test)

        logging.info('Data cleaned')

    if stem:
        data_source_train = stem_data(data_source_train)
        data_source_dev = stem_data(data_source_dev)

        data_target_train = stem_data(data_target_train)
        data_target_dev = stem_data(data_target_dev)

        data_tune_train = stem_data(data_tune_train)
        data_tune_dev = stem_data(data_tune_dev)

        data_clinical_test = stem_data(data_clinical_test)

        logging.info('Data stemmed')

    if use_token_level_attention:
        data_source_train = create_token_cuis(data_source_train)
        data_source_dev = create_token_cuis(data_source_dev)
        data_target_dev = create_token_cuis(data_target_dev)

        data_clinical_test = create_token_cuis(data_clinical_test)

    word_vectors_replacement = None
    if word_vectors_replace_cui != '':
        word_vectors_replacement_filename = get_word_vectors_filename(word_vectors_replace_cui)
        word_vectors_replacement = load_pickle(word_vectors_replacement_filename)
        logging.info('Replacements word vectors loaded: %s', word_vectors_replacement_filename.name)

        target_cuis = set(word_vectors_replacement.keys())
        logging.info('Target CUIs: %s', len(target_cuis))

        data_source_train = replace_cui_data(data_source_train, target_cuis)
        data_source_dev = replace_cui_data(data_source_dev, target_cuis)

        data_target_train = replace_cui_data(data_target_train, target_cuis)
        data_target_dev = replace_cui_data(data_target_dev, target_cuis)

        data_tune_train = replace_cui_data(data_tune_train, target_cuis)
        data_tune_dev = replace_cui_data(data_tune_dev, target_cuis)

        data_clinical_test = replace_cui_data(data_clinical_test, target_cuis)

        logging.info('CUIs replaced')

    if downsample_source != 0:
        # downsample train and dev sets to the size of the clinical dataset
        nb_clinical_train = 11232
        nb_clinical_dev = 1395

        data_source_train = downsample_data(data_source_train, nb_needed=nb_clinical_train)
        data_source_dev = downsample_data(data_source_dev, nb_needed=nb_clinical_dev)

    # create tokenizer and vocabulary
    sentences_train = data_source_train['premise'] + data_source_train['hypothesis']
    if data_tune_train is not None:
        sentences_train += data_tune_train['premise'] + data_tune_train['hypothesis']

    tokenizer = Tokenizer(lower=lowercase, filters='')
    tokenizer.fit_on_texts(sentences_train)

    # create data matrices
    m_source_train = create_data_matrices(tokenizer, data_source_train, max_len, padding)
    m_source_dev = create_data_matrices(tokenizer, data_source_dev, max_len, padding)
    logging.info('Source: %s - train: %s, %s, %s, dev: %s, %s, %s', genre_source,
                 m_source_train['premise'].shape, m_source_train['hypothesis'].shape, m_source_train['label'].shape,
                 m_source_dev['premise'].shape, m_source_dev['hypothesis'].shape, m_source_dev['label'].shape)

    m_tune_train = None
    m_tune_dev = None
    if data_tune_train is not None:
        m_tune_train = create_data_matrices(tokenizer, data_tune_train, max_len, padding)
        m_tune_dev = create_data_matrices(tokenizer, data_tune_dev, max_len, padding)
        logging.info('Tune: %s - train: %s, %s, %s, dev: %s, %s, %s', genre_tune,
                     m_tune_train['premise'].shape, m_tune_train['hypothesis'].shape, m_tune_train['label'].shape,
                     m_tune_dev['premise'].shape, m_tune_dev['hypothesis'].shape, m_tune_dev['label'].shape)

    m_target_train = None
    m_target_dev = None
    if data_target_train is not None:
        m_target_train = create_data_matrices(tokenizer, data_target_train, max_len, padding)
        m_target_dev = create_data_matrices(tokenizer, data_target_dev, max_len, padding)
        logging.info('Target: %s - train: %s, %s, %s, dev: %s, %s, %s', genre_target,
                     m_target_train['premise'].shape, m_target_train['hypothesis'].shape, m_target_train['label'].shape,
                     m_target_dev['premise'].shape, m_target_dev['hypothesis'].shape, m_target_dev['label'].shape)

    else:
        m_target_dev = m_source_dev  # target domain was not specified - use the dev set of the source domain
        data_target_dev = data_source_dev
        logging.info('Target: %s - dev: %s, %s, %s', genre_source,
                     m_target_dev['premise'].shape, m_target_dev['hypothesis'].shape, m_target_dev['label'].shape)

    m_clinical_test = create_data_matrices(tokenizer, data_clinical_test, max_len, padding)
    logging.info('Clinical test: %s, %s, %s',
                 m_clinical_test['premise'].shape, m_clinical_test['hypothesis'].shape, m_clinical_test['label'].shape)

    # create embedding matrix
    if word_vectors_type != 'random':
        word_vectors_filename = get_word_vectors_filename(word_vectors_type)
        word_vectors = load_pickle(word_vectors_filename)
        logging.info('Word vectors loaded: %s', word_vectors_filename.name)

        if word_vectors_replacement is not None:
            word_vectors.update(word_vectors_replacement)

    else:
        random_vectors_params = (-0.5, 0.5, 300,)
        word_vectors = {}
        for token in tokenizer.word_index.keys():
            word_vectors[token] = np.random.uniform(*random_vectors_params)

        logging.info('Random vectors created: %s', random_vectors_params)

    W_emb = create_embedding_matrix(word_vectors, tokenizer.word_index)

    id_to_token = {i: t for t, i in tokenizer.word_index.items()}
    logging.info('Id to token: %s', len(id_to_token))

    if word_vectors_replace_cui != '' or use_token_level_attention:
        concepts_graph = nx.read_gpickle(str(UMLS_CONCEPTS_GRAPH_FILENAME))
        logging.info('UMLS concepts graph: %s', len(concepts_graph))

        # create UMLS-based attention
        if use_token_level_attention:
            att_source_train = create_umls_attention(
                m_source_train, id_to_token, concepts_graph,
                use_token_level_attention, data_source_train['premise_token_cuis'],
                data_source_train['hypothesis_token_cuis'])
            att_source_dev = create_umls_attention(
                m_source_dev, id_to_token, concepts_graph,
                use_token_level_attention, data_source_dev['premise_token_cuis'],
                data_source_dev['hypothesis_token_cuis'])
            att_target_dev = create_umls_attention(
                m_target_dev, id_to_token, concepts_graph,
                use_token_level_attention, data_target_dev['premise_token_cuis'],
                data_target_dev['hypothesis_token_cuis'])

            att_clinical_test = create_umls_attention(
                m_clinical_test, id_to_token, concepts_graph,
                use_token_level_attention, data_clinical_test['premise_token_cuis'],
                data_clinical_test['hypothesis_token_cuis'])

            m_source_train.update(att_source_train)
            m_source_dev.update(att_source_dev)
            m_target_dev.update(att_target_dev)

            m_clinical_test.update(att_clinical_test)

        # create memory
        if not use_token_level_attention:
            memory_source_train = create_memory_matrix(m_source_train, id_to_token, concepts_graph, word_vectors,
                                                       use_token_level_attention)
            memory_source_dev = create_memory_matrix(m_source_dev, id_to_token, concepts_graph, word_vectors,
                                                     use_token_level_attention)
            memory_target_dev = create_memory_matrix(m_target_dev, id_to_token, concepts_graph, word_vectors,
                                                     use_token_level_attention)

            memory_clinical_test = create_memory_matrix(m_clinical_test, id_to_token, concepts_graph, word_vectors,
                                                        use_token_level_attention)

            m_source_train.update(memory_source_train)
            m_source_dev.update(memory_source_dev)
            m_target_dev.update(memory_target_dev)

            m_clinical_test.update(memory_clinical_test)

    # use WordNet attention
    if genre_source != 'clinical' and word_vectors_replace_cui == '' and use_umls_attention:
        att_source_train = create_wordnet_attention(m_source_train, id_to_token)
        att_source_dev = create_wordnet_attention(m_source_dev, id_to_token)
        att_target_dev = create_wordnet_attention(m_target_dev, id_to_token)

        att_clinical_test = create_wordnet_attention(m_clinical_test, id_to_token)

        m_source_train.update(att_source_train)
        m_source_dev.update(att_source_dev)
        m_target_dev.update(att_target_dev)

        m_clinical_test.update(att_clinical_test)

    # save tokenizer and embeddings matrix for demo server
    save_pickle(DATA_DIR / 'tokenizer_{}_{}.pickled'.format(genre_source, genre_tune), tokenizer)
    save_pickle(DATA_DIR / 'embeddings_{}_{}.pickled'.format(genre_source, genre_tune), W_emb)

    return m_source_train, m_source_dev, m_tune_train, m_tune_dev, m_target_train, m_target_dev, m_clinical_test, W_emb


def create_model(model_name, model_params):
    if model_name == 'PyTorchSimpleModel':
        model_class = PyTorchSimpleModel
    elif model_name == 'PyTorchInferSentModel':
        model_class = PyTorchInferSentModel
    elif model_name == 'PyTorchESIMModel':
        model_class = PyTorchESIMModel

    elif model_name == 'PyTorchMultiTargetSimpleModel':
        model_class = PyTorchMultiTargetSimpleModel
    elif model_name == 'PyTorchMultiTargetInferSentModel':
        model_class = PyTorchMultiTargetInferSentModel
    elif model_name == 'PyTorchMultiTargetESIMModel':
        model_class = PyTorchMultiTargetESIMModel
    else:
        raise ValueError('Model class {} is unknown'.format(model_name))

    model = model_class(**model_params)
    logging.info('Model created: %s', type(model).__name__)

    model.build()

    return model


def train_model(model, m_source_train, m_source_dev, m_tune_train, m_tune_dev, m_target_train, m_target_dev,
                genre_source, genre_target, genre_tune, lambda_multi_task, uniform_batches,
                batch_size, epochs, verbose):
    if lambda_multi_task != -1:
        logging.info('Multi-task learning: %s - %s, lambda: %s, uniform: %s',
                     genre_source, genre_target, lambda_multi_task, uniform_batches)

        model.train_sampled(m_source_train, m_target_train, data_dev=m_target_dev,
                            lambda_multi=lambda_multi_task, uniform_batches=uniform_batches,
                            batch_size=batch_size, epochs=epochs, verbose=verbose)

    else:
        if isinstance(model, PyTorchMultiTargetBaseModel):
            logging.info('Training on source - multi target: %s, %s samples', genre_source,
                         len(m_source_train['premise']))
            model.train_source(m_source_train, data_dev=m_source_dev,
                               batch_size=batch_size, epochs=epochs, verbose=verbose)
        else:
            logging.info('Training on source: %s, %s samples', genre_source, len(m_source_train['premise']))
            model.train(m_source_train, data_dev=m_source_dev,
                        batch_size=batch_size, epochs=epochs, verbose=verbose)

        if m_tune_train is not None:
            if isinstance(model, PyTorchMultiTargetBaseModel):
                logging.info('Training on tune - multi target: %s, %s samples', genre_tune,
                             len(m_tune_train['premise']))
                model.train_target(m_tune_train, data_dev=m_tune_dev,
                                   batch_size=batch_size, epochs=epochs, verbose=verbose)
            else:
                logging.info('Training on tune: %s, %s samples', genre_tune, len(m_tune_train['premise']))
                model.train(m_tune_train, data_dev=m_tune_dev,
                            batch_size=batch_size, epochs=epochs, verbose=verbose)


def evaluate_model_on_sets(model, m_source_train, m_source_dev, m_tune_train, m_tune_dev, m_target_dev,
                           m_clinical_test=None, batch_size=512):
    result = {}

    eval_sets = [
        ('source_train', m_source_train),
        ('source_dev', m_source_dev),
        ('target_dev', m_target_dev),
    ]

    if m_clinical_test is not None:
        eval_sets.append(('clinical_test', m_clinical_test))

    if m_tune_train is not None:
        eval_sets.append(('tune_train', m_tune_train))
        eval_sets.append(('tune_dev', m_tune_dev))

    for name, eval_set in eval_sets:
        loss, acc = model.evaluate(eval_set, batch_size=batch_size)
        result['loss_' + name] = loss
        result['acc_' + name] = acc

    return result


@ex.config
def config():
    model_class = 'PyTorchInferSentModel'  # class name of the model to run. See the `create_model` function for the available models
    max_len = 50  # max sentence length
    lowercase = False  # lowercase input data or nor
    clean = False  # remove punctuation etc or not
    stem = False  # do stemming to not
    word_vectors_type = 'glove'  # word vectors - see the `WORD_VECTORS_FILENAME` in `config.py` for details
    word_vectors_replace_cui = ''  # filename with retorifitted embeddings for CUIs, eg cui.glove.cbow_most_common.CHD-PAR.SNOMEDCT_US.retrofitted.pkl
    downsample_source = 0  # down sample the source domain data to the size of the MedNLI

    # transfer learning settings
    genre_source = 'clinical'  # source domain for transfer learning. target='' and tune='' - no transfer
    genre_target = ''  # target domain - always MedNLI in case of experiemnts in the paper
    genre_tune = ''  # fine-tuning domain
    lambda_multi_task = -1  # whether to use dynamically sampled batches from different domains or not.
    uniform_batches = True  # a batch will contain samples from just one domain

    rnn_size = 300  # size of LSTM
    rnn_cell = 'LSTM'  # LSTM is used in the experiments in the paper
    regularization = 0.000001  # regularization strength
    dropout = 0.5  # dropout
    hidden_size = 300  # size of the hidden fully-connected layers
    trainable_embeddings = False  # train embeddings or not

    # knowledge-based attention
    # set both to true to reproduce the token-level UMLS attention used in the paper
    use_umls_attention = False  # whether to use the knowledge-based attention or not
    use_token_level_attention = False  # use CUIs or separate tokens for attention

    batch_size = 512  # batch size
    epochs = 40  # number of epochs for training
    learning_rate = 0.001  # learning rate for the Adam optimizer
    training_loop_mode = 'best_loss'  # best_loss or best_acc - the model will be saved on the base loss or accuracy on the validation set correspondingly

    checkpoint_file_prefix = '{}_{}_{}_{}_{}_{}'.format(
        model_class, max_len, word_vectors_type, genre_source, genre_target, genre_tune
    )

    verbose = 1


@ex.main
def main(model_class, genre_source, genre_target, genre_tune, lowercase, stem, clean, downsample_source,
         lambda_multi_task, uniform_batches, max_len, word_vectors_type, word_vectors_replace_cui,
         use_umls_attention, use_token_level_attention, checkpoint_file_prefix, epochs, batch_size, verbose, _config):
    post_padding_models = {
        'PyTorchESIMModel', 'PyTorchMultiTargetESIMModel', 'PyTorchInferSentModel', 'PyTorchMultiTargetInferSentModel',
    }
    if model_class in post_padding_models:
        padding = 'post'
    else:
        padding = 'pre'
    logging.info('Padding: %s', padding)

    # create data matrices for all genres
    m_source_train, m_source_dev, m_tune_train, m_tune_dev, m_target_train, m_target_dev, m_clinical_test, W_emb = process_data(
        genre_source, genre_target, genre_tune, max_len, lowercase, stem, clean, downsample_source,
        word_vectors_type, word_vectors_replace_cui, use_umls_attention, use_token_level_attention, padding
    )

    if not MODELS_WEIGHTS_DIR.exists():
        MODELS_WEIGHTS_DIR.mkdir()
    _, checkpoint_filename = tempfile.mkstemp(dir=str(MODELS_WEIGHTS_DIR),
                                              prefix=checkpoint_file_prefix + '.', suffix='.h5')

    model_params = {
        'W_emb': W_emb,
        'max_len': max_len,
        'checkpoint_filename': checkpoint_filename,
    }
    model_params.update(_config)

    model = create_model(model_class, model_params)

    # train the model
    train_model(
        model, m_source_train, m_source_dev, m_tune_train, m_tune_dev, m_target_train, m_target_dev,
        genre_source, genre_target, genre_tune, lambda_multi_task, uniform_batches, batch_size, epochs, verbose
    )

    # evaluate model
    model.load_weights(checkpoint_filename)

    result = evaluate_model_on_sets(
        model, m_source_train, m_source_dev, m_tune_train, m_tune_dev, m_target_dev, m_clinical_test, batch_size)

    result['checkpoint_filename'] = checkpoint_filename
    # ex.add_artifact(checkpoint_filename)

    if 'acc_target_dev' in result:
        logging.info('Target accuracy: %.3f', result['acc_target_dev'])

    if 'acc_clinical_test' in result:
        logging.info('Clinical test accuracy: %.3f', result['acc_clinical_test'])

    return result


if __name__ == '__main__':
    ex.run_commandline()
