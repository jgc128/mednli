import logging
import itertools
from collections import defaultdict, Counter

import networkx as nx
import numpy as np
from sacred import Experiment

from config import UMLS_CONCEPTS_GRAPH_FILENAME, PROCESSED_DATA_DIR, \
    PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE, WORD_VECTORS_FILENAME, RETROFITTING_DIR
from utils.data import load_single_genre_data
from utils.io import load_pickle, save_pickle

ex = Experiment('create_retrofitting_data')


def save_retrofitting_lexicon_list(filename, graph, selected_relations, selected_sources):
    retrofitting_list = defaultdict(list)
    target_cuis = []

    for u, v, d in graph.edges_iter(data=True):
        rel = d['rel']
        source = d['sab']

        if len(selected_relations) == 0 or 'all' in selected_relations or rel in selected_relations:
            if len(selected_sources) == 0 or 'all' in selected_sources or source in selected_sources:
                retrofitting_list[u].append(v)

                target_cuis.append(u)
                target_cuis.append(v)

    target_cuis = set(target_cuis)
    logging.info('Lexicon: %s', len(retrofitting_list))
    logging.info('CUIs: %s', len(target_cuis))

    nb_relations_before = 0
    nb_relations_after = 0
    with open(str(filename), 'w') as f:
        for target_cui, related_cuis in retrofitting_list.items():
            nb_relations_before += len(related_cuis)
            related_cuis = set(related_cuis)
            nb_relations_after += len(related_cuis)

            row = '{} {}\n'.format(target_cui, ' '.join(related_cuis))
            f.write(row)

    logging.info('Lexicon saved: %s, %s -> %s', filename.name, nb_relations_before, nb_relations_after)

    return target_cuis


def save_embeddings(filename, data_train, data_dev, word_vectors, target_cuis, mode):
    # find with which words concepts are presented in the data
    concepts_tokens = defaultdict(list)
    data_all = [
        zip(data_train['premise'], data_train['premise_concepts']),
        zip(data_train['hypothesis'], data_train['hypothesis_concepts']),
    ]
    for i, (sentence, concepts) in enumerate(itertools.chain.from_iterable(data_all)):
        for concept in concepts:
            cui = concept['cui']

            if cui not in target_cuis:
                continue

            pos_info = concept['pos_info']

            tokens = [sentence[p[0]:p[1]] for p in pos_info]
            concepts_tokens[cui].extend(tokens)

    logging.info('Concepts mode: %s', mode)
    logging.info('Concepts: %s', len(concepts_tokens))

    # filter out concepts without a representation with a single token,
    # and the rest is the same as the cbow_most_common mode
    if mode == 'single_token':
        concepts_tokens = {
            concept: [t for t in tokens_list if not ' ' in t]
            for concept, tokens_list in concepts_tokens.items()
        }
        concepts_tokens = {c: t for c, t in concepts_tokens.items() if len(t) > 0}
        logging.info('Concepts single tokens: %s', len(concepts_tokens))

    if mode == 'cbow_all':
        concepts_tokens = {
            cui: [tok for tokens in tokens_list for tok in tokens.split()]
            for cui, tokens_list in concepts_tokens.items()
        }
        concepts_tokens = {cui: set(tokens) for cui, tokens in concepts_tokens.items()}

    elif mode == 'single_most_common':
        concepts_tokens = {
            cui: [tok for tokens in tokens_list for tok in tokens.split()]
            for cui, tokens_list in concepts_tokens.items()
        }
        concepts_tokens_counter = {cui: Counter(tokens) for cui, tokens in concepts_tokens.items()}
        concepts_tokens = {}
        for concept, tokens_counts in concepts_tokens_counter.items():
            # there might be several tokens with the same frequency - take the longest one in this case
            _, nb_most_common = tokens_counts.most_common(1)[0]
            tokens = [t for t, c in tokens_counts.most_common() if c == nb_most_common]
            tokens = sorted(tokens, key=lambda x: len(x), reverse=True)
            concepts_tokens[concept] = tokens[:1]

    elif mode == 'cbow_most_common' or mode == 'single_token':
        concepts_tokens_counter = {cui: Counter(tokens) for cui, tokens in concepts_tokens.items()}

        concept_tokens = {}
        for concept, tokens_counts in concepts_tokens_counter.items():
            # add first most common that have at least one embedding
            for tokens, counts in tokens_counts.most_common():
                tokens = tokens.split(' ')
                if any([t in word_vectors for t in tokens]):
                    concept_tokens[concept] = tokens
                    break

    else:
        raise ValueError('Unknown mode: {}'.format(mode))

    logging.info('Concepts tokens: %s', len(concepts_tokens))

    # create a word vectors for each CUI as an average of embeddings
    cuis_embeddings = {}
    for cui, tokens in concepts_tokens.items():
        cui_embeds = []
        for token in tokens:
            if token in word_vectors:
                cui_embeds.append(word_vectors[token])

        if len(cui_embeds) > 0:
            cuis_embeddings[cui] = np.mean(cui_embeds, axis=0)

    logging.info('Concepts with embeddings: %s', len(cuis_embeddings))

    del word_vectors
    if filename.suffix == '.txt':
        # save embeddings in the retrofitting format
        with open(str(filename), 'w') as f:
            for cui, embeddings in cuis_embeddings.items():
                row = '{} {}\n'.format(cui, ' '.join(embeddings.astype(str)))
                f.write(row)
    else:
        save_pickle(filename, cuis_embeddings)

    logging.info('Embeddings saved: %s', filename.name)


@ex.config
def config():
    word_vectors_type = 'glove'
    mode = 'cbow_most_common'  # cbow_most_common single_most_common cbow_all
    selected_relations = 'all'  # RO-RQ-PAR-CHD-RN-RB-RU-RL
    selected_sources = 'SNOMEDCT_US'


@ex.main
def main(word_vectors_type, selected_relations, selected_sources, mode):
    # load graph
    graph = nx.read_gpickle(str(UMLS_CONCEPTS_GRAPH_FILENAME))

    # load word vectors
    word_vectors_filename = WORD_VECTORS_FILENAME[word_vectors_type]
    word_vectors = load_pickle(word_vectors_filename)
    logging.info('Word vectors loaded: %s', word_vectors_filename)

    data_train, data_dev = load_single_genre_data(
        PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE, genre='clinical')
    logging.info('Data train: %s, data_dev: %s', len(data_train['premise']), len(data_dev['premise']))

    # save lexicon list
    selected_relations = set(selected_relations.split('-'))
    logging.info('Selected relations: %s', selected_relations)

    selected_sources = set(selected_sources.split('-'))
    logging.info('Selected sources: %s', selected_sources)

    lexicon_filename = RETROFITTING_DIR / 'lexicon.{}.{}.txt'.format(
        '-'.join(sorted(selected_relations)), '-'.join(sorted(selected_sources)))
    target_cuis = save_retrofitting_lexicon_list(lexicon_filename, graph, selected_relations, selected_sources)

    # save embeddings
    embeddings_filename = RETROFITTING_DIR / 'cui.{}.{}.{}.{}.pkl'.format(
        word_vectors_type, mode, '-'.join(sorted(selected_relations)), '-'.join(sorted(selected_sources))
    )
    save_embeddings(embeddings_filename, data_train, data_dev, word_vectors, target_cuis, mode)


if __name__ == '__main__':
    ex.run_commandline()
