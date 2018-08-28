import logging
import itertools
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from nltk.corpus import wordnet as wn
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from tqdm import tqdm
from sacred import Experiment

from config import SNLI_TRAIN_FILENAME, SNLI_DEV_FILENAME, MULTINLI_TRAIN_FILENAME, MULTINLI_DEV_FILENAME, \
    MLI_TRAIN_FILENAME, MLI_DEV_FILENAME, WORD_VECTORS_FILENAME, PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_TEMPLATE, \
    MLI_TEST_FILENAME, PROCESSED_DATA_TEST_FILENAME_TEMPLATE
from utils.data import format_processed_filename
from utils.io import save_pickle
from utils.nli import read_nli_data, LABELS, get_tokens_from_parse, GENRES
from utils.text import get_tokens_cuis

ex = Experiment('train_model')


def load_data(limit=None, load_test=False):
    """Load SNLI, MultiNLI and MLI datasets into train/dev DataFrames"""
    data_snli_dev, data_snli_train = None, None
    data_multinli_train, data_multinli_dev = None, None
    data_mli_train, data_mli_dev = None, None
    data_mli_test = None

    if SNLI_TRAIN_FILENAME.exists():
        data_snli_train = read_nli_data(SNLI_TRAIN_FILENAME, set_genre='snli', limit=limit)
        data_snli_dev = read_nli_data(SNLI_DEV_FILENAME, set_genre='snli', limit=limit)
        logging.info('SNLI: train - %s, dev - %s', data_snli_train.shape, data_snli_dev.shape)

    if MULTINLI_TRAIN_FILENAME.exists():
        data_multinli_train = read_nli_data(MULTINLI_TRAIN_FILENAME, limit=limit)
        data_multinli_dev = read_nli_data(MULTINLI_DEV_FILENAME, limit=limit)
        logging.info('MultiNLI: train - %s, dev - %s', data_multinli_train.shape, data_multinli_dev.shape)

    if MLI_TRAIN_FILENAME.exists():
        data_mli_train = read_nli_data(MLI_TRAIN_FILENAME, set_genre='clinical', limit=limit)
        data_mli_dev = read_nli_data(MLI_DEV_FILENAME, set_genre='clinical', limit=limit)
        logging.info('MLI: train - %s, dev - %s', data_mli_train.shape, data_mli_dev.shape)

    if load_test:
        data_mli_test = read_nli_data(MLI_TEST_FILENAME, set_genre='clinical', limit=limit)

    # Drop columns that are presented not in all datasets
    columns_to_drop = ['captionID', 'promptID', 'annotator_labels']
    for d in [data_snli_dev, data_snli_train, data_multinli_train, data_multinli_dev, data_mli_train, data_mli_dev,
              data_mli_test]:
        if d is not None:
            d.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    # concatenate all data together
    data_train = pd.concat([data_snli_train, data_multinli_train, data_mli_train], axis=0)
    data_dev = pd.concat([data_snli_dev, data_multinli_dev, data_mli_dev], axis=0)

    data_train.set_index('genre', inplace=True)
    data_dev.set_index('genre', inplace=True)

    if data_mli_test is not None:
        data_mli_test.set_index('genre', inplace=True)

    if not load_test:
        return data_dev, data_train
    else:
        return data_dev, data_train, data_mli_test


def tokenize_data(data):
    labels = data['gold_label'].map(LABELS).tolist()
    premise_tokens = data['sentence1_binary_parse'].map(get_tokens_from_parse).tolist()
    hypothesis_tokens = data['sentence2_binary_parse'].map(get_tokens_from_parse).tolist()

    return premise_tokens, hypothesis_tokens, labels


def create_token_ids_matrix(tokenizer, sequences, max_len, padding):
    tokens_ids = tokenizer.texts_to_sequences(sequences)

    # there might be zero len sequences - fix it by putting a random token there (or id 1 in the worst case)
    tokens_ids_flattened = list(itertools.chain.from_iterable(tokens_ids))
    max_id = max(tokens_ids_flattened) if len(tokens_ids_flattened) > 0 else -1
    for i in range(len(tokens_ids)):
        if len(tokens_ids[i]) == 0:
            id_to_put = np.random.randint(1, max_id) if max_id != -1 else 1
            tokens_ids[i].append(id_to_put)

    tokens_ids = pad_sequences(tokens_ids, maxlen=max_len, padding=padding)
    return tokens_ids


def create_data_matrices(tokenizer, data, max_len, padding):
    premise = create_token_ids_matrix(tokenizer, data['premise'], max_len, padding)
    hypothesis = create_token_ids_matrix(tokenizer, data['hypothesis'], max_len, padding)
    label = np.array(data['label'])

    m_data = {
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
    }
    return m_data


def create_embedding_matrix(word_vectors, token2id):
    embedding_dim = list(word_vectors.values())[0].shape[0]
    nb_tokens = len(token2id)

    W_emb = np.zeros((nb_tokens + 1, embedding_dim))

    # a vector for unknown words (that are not in glove vocab)
    unk_vector = np.random.uniform(-0.5, 0.5, (embedding_dim,))

    nb_unk = 0
    for token, idx in token2id.items():
        if token in word_vectors:
            W_emb[idx] = word_vectors[token]
        else:
            W_emb[idx] = unk_vector
            nb_unk += 1

    logging.info('Embedding matrix created: %s, unknown tokens: %s', W_emb.shape, nb_unk)
    return W_emb


def create_memory_matrix(data, id_to_token, concepts_graph, word_vectors, use_token_level_attention):
    premise = data['premise']
    hypothesis = data['hypothesis']

    embeddings_dim = next(iter(word_vectors.values())).shape[0]

    # first, create a dict for each sample
    memories = []
    for i, (prem, hyp) in enumerate(zip(premise, hypothesis)):
        memory = defaultdict(set)
        all_tokens_ids = set(itertools.chain(prem, hyp))
        all_tokens = [id_to_token[t_id] for t_id in all_tokens_ids if t_id != 0]
        all_concepts = [c for c in all_tokens if c in concepts_graph and c in word_vectors]

        for concept in all_concepts:
            for related_concept, relations_dict in concepts_graph[concept].items():
                if related_concept not in word_vectors:
                    continue

                for relation_id, relation in relations_dict.items():
                    is_needed_relation = relation['rel'] == 'PAR' or relation['rel'] == 'CHD'
                    is_needed_source = relation['sab'] == 'SNOMEDCT_US'
                    if is_needed_relation and is_needed_source:
                        memory[concept].add(related_concept)

        memories.append(memory)

    mem_size_max = max(len(m) for m in memories)
    mem_size_avg = np.mean([len(m) for m in memories])
    logging.info('Memory set created: max size %s, avg size: %s', mem_size_max, mem_size_avg)

    # second, create key value matrices
    m_memory_keys = np.zeros((len(data['premise']), mem_size_max, embeddings_dim), dtype=np.float32)
    m_memory_values = np.zeros((len(data['premise']), mem_size_max, embeddings_dim), dtype=np.float32)
    for i, memory_dict in enumerate(memories):
        for j, (concept, related_concepts) in enumerate(memory_dict.items()):
            m_memory_keys[i, j, :] = word_vectors[concept]

            mem_vals = [word_vectors[rc] for rc in related_concepts]
            m_memory_values[i, j, :] = np.mean(mem_vals, axis=0)

    logging.info('Memory matrices created: %s, %s', m_memory_keys.shape, m_memory_values.shape)

    m_mem = {
        'memory_key': m_memory_keys,
        'memory_value': m_memory_values,
    }
    return m_mem


def create_umls_attention(data, id_to_token, concepts_graph, use_token_level_attention, premise_concepts,
                          hypothesis_concepts):
    # process the graph to extract only needed relations
    needed_relations = nx.MultiDiGraph()
    for u, v, d in concepts_graph.edges_iter(data=True):
        rel = d['rel']
        source = d['sab']

        is_needed_source = source == 'SNOMEDCT_US'  # 'SNOMEDCT_US'
        is_needed_relation = True  # rel == 'PAR' or rel == 'CHD'
        if is_needed_relation and is_needed_source:
            needed_relations.add_edge(u, v, rel=rel, sab=source)

    logging.info('Needed relations extracted: {}'.format(len(needed_relations)))

    # create attention matrices
    premise = data['premise']
    hypothesis = data['hypothesis']

    nb_samples = premise.shape[0]
    max_len_premise = premise.shape[1]
    max_len_hypothesis = hypothesis.shape[1]

    no_path_val = 999
    path_lengths = np.full((nb_samples, max_len_premise, max_len_hypothesis), no_path_val, dtype=np.float32)
    for k, (prem, hyp) in enumerate(zip(premise, hypothesis)):
        for i in range(max_len_premise):
            if prem[i] == 0:
                continue

            if use_token_level_attention:
                if i not in premise_concepts[k]:
                    continue

                token_cuis_prem = premise_concepts[k][i]
                token_cuis_filtered_prem = [c for c in token_cuis_prem if c['cui'] in needed_relations]
                if len(token_cuis_filtered_prem) == 0:
                    continue

            else:
                token_premise = id_to_token[prem[i]]
                if token_premise not in needed_relations:
                    continue

            for j in range(max_len_hypothesis):
                if hyp[j] == 0:
                    continue

                if use_token_level_attention:
                    if j not in hypothesis_concepts[k]:
                        continue

                    token_cuis_hyp = hypothesis_concepts[k][j]
                    token_cuis_filtered_hyp = [c for c in token_cuis_hyp if c['cui'] in needed_relations]
                    if len(token_cuis_filtered_hyp) == 0:
                        continue

                else:
                    token_hypothesis = id_to_token[hyp[j]]
                    if token_hypothesis not in needed_relations:
                        continue

                if use_token_level_attention:
                    path_ph = [
                        get_path_len(needed_relations, cui_h['cui'], cui_p['cui'], no_path_val)
                        for cui_h, cui_p in itertools.product(token_cuis_filtered_prem, token_cuis_filtered_hyp)
                    ]
                    path_hp = [
                        get_path_len(needed_relations, cui_p['cui'], cui_h['cui'], no_path_val)
                        for cui_h, cui_p in itertools.product(token_cuis_filtered_prem, token_cuis_filtered_hyp)
                    ]
                    path_len = min(itertools.chain(path_ph, path_hp))
                else:
                    path_ph_len = get_path_len(needed_relations, token_premise, token_hypothesis, no_path_val)
                    path_hp_len = get_path_len(needed_relations, token_hypothesis, token_premise, no_path_val)
                    path_len = min(path_ph_len, path_hp_len)

                path_lengths[k, i, j] = path_len

    logging.info('Path matrix: %s', path_lengths.shape)

    # convert path lengths to attention - the shorter path is, the more attention should be placed
    no_path = path_lengths == no_path_val
    path_lengths[no_path] = 0
    max_path_len = path_lengths.max()
    path_lengths /= max_path_len
    path_lengths = 1 - path_lengths
    path_lengths[no_path] = 0

    logging.info('Attention created: %s', path_lengths.shape)

    m_attention = {
        'attention': path_lengths,
    }
    return m_attention


def get_path_len(graph, cui1, cui2, no_path_val):
    try:
        path_ph = nx.shortest_path(graph, cui1, cui2)
        path_ph_len = len(path_ph) - 1
    except nx.NetworkXNoPath as e:
        path_ph_len = no_path_val

    return path_ph_len


def create_wordnet_attention(data, id_to_token):
    def _get_synset(token):
        synsets = wn.synsets(token)
        if len(synsets) > 0:
            return synsets[0]
        else:
            return None

    # create attention matrices
    premise = data['premise']
    hypothesis = data['hypothesis']

    nb_samples = premise.shape[0]
    max_len_premise = premise.shape[1]
    max_len_hypothesis = hypothesis.shape[1]

    # cache path len between synsents
    id_to_synset = {i: _get_synset(t) for i, t in id_to_token.items()}
    id_to_synset = {i: s for i, s in id_to_synset.items() if s is not None}
    logging.info('Tokens: %s, synsets: %s', len(id_to_token), len(id_to_synset))

    # create path len matrix
    no_path_val = 999
    path_lengths_cache = {}
    path_lengths = np.full((nb_samples, max_len_premise, max_len_hypothesis), no_path_val, dtype=np.float32)
    for k, (prem, hyp) in enumerate(zip(premise, hypothesis)):
        for i in range(max_len_premise):
            if prem[i] == 0:
                continue

            token_id_premise = prem[i]

            for j in range(max_len_hypothesis):
                if hyp[j] == 0:
                    continue

                token_id_hypothesis = hyp[j]

                if token_id_premise not in id_to_synset or token_id_hypothesis not in id_to_synset:
                    continue

                if (token_id_premise, token_id_hypothesis) not in path_lengths_cache:
                    synset_premise = id_to_synset[token_id_premise]
                    synset_hypothesis = id_to_synset[token_id_hypothesis]

                    path_len = synset_premise.shortest_path_distance(synset_hypothesis, True) or no_path_val

                    path_lengths_cache[(token_id_premise, token_id_hypothesis)] = path_len
                    path_lengths_cache[(token_id_hypothesis, token_id_premise)] = path_len

                path_len = path_lengths_cache[(token_id_premise, token_id_hypothesis)]
                path_lengths[k, i, j] = path_len

    logging.info('Path matrix: %s', path_lengths.shape)

    # convert path lengths to attention - the shorter path is, the more attention should be placed
    no_path = path_lengths == no_path_val
    path_lengths[no_path] = 0
    max_path_len = path_lengths.max()
    path_lengths /= max_path_len
    path_lengths = 1 - path_lengths
    path_lengths[no_path] = 0

    logging.info('Attention created: %s', path_lengths.shape)

    m_attention = {
        'attention': path_lengths,
    }
    return m_attention


def create_token_cuis(data):
    if data is None:
        return None

    premise = data['premise']
    hypothesis = data['hypothesis']

    premise_concepts = data['premise_concepts']
    hypothesis_concepts = data['hypothesis_concepts']

    premise_token_cuis = [
        get_tokens_cuis(sentence, sentence_cuis)
        for sentence, sentence_cuis in zip(premise, premise_concepts)
    ]

    hypothesis_token_cuis = [
        get_tokens_cuis(sentence, sentence_cuis)
        for sentence, sentence_cuis in zip(hypothesis, hypothesis_concepts)
    ]

    data['premise_token_cuis'] = premise_token_cuis
    data['hypothesis_token_cuis'] = hypothesis_token_cuis

    return data


@ex.main
def main():
    # load SNLI, MultiNLI and MLI datasets
    data_dev, data_train = load_data()
    logging.info('Data: train - %s, dev - %s', data_train.shape, data_dev.shape)

    if not PROCESSED_DATA_DIR.exists():
        PROCESSED_DATA_DIR.mkdir()

    for genre in GENRES:
        if genre not in data_train.index:
            continue

        genre_train = data_train.loc[genre]
        genre_dev = data_dev.loc[genre]
        logging.info('Genre: %s, train: %s, dev: %s', genre, genre_train.shape, genre_dev.shape)

        tokenized_train = tokenize_data(genre_train)
        tokenized_dev = tokenize_data(genre_dev)

        # save all the data into a numpy file
        filename = format_processed_filename(PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_TEMPLATE, genre=genre)
        save_pickle(filename, (tokenized_train, tokenized_dev))


@ex.command
def process_test():
    # load the test data
    _, _, data_mli_test = load_data(load_test=True)
    logging.info('Data: %s', data_mli_test.shape)

    for genre in ['clinical']:
        genre_test = data_mli_test.loc[genre]
        logging.info('Genre: %s, test: %s', genre, genre_test.shape)

        tokenized_test = tokenize_data(genre_test)

        # save all the data into a numpy file
        filename = format_processed_filename(PROCESSED_DATA_DIR, PROCESSED_DATA_TEST_FILENAME_TEMPLATE, genre=genre)
        save_pickle(filename, tokenized_test)


if __name__ == '__main__':
    ex.run_commandline()
