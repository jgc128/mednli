import os

from pymetamap import MetaMap, ConceptMMI

from config import METAMAP_BINARY_PATH, PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_TEMPLATE, \
    PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE, PROCESSED_DATA_TEST_FILENAME_TEMPLATE, \
    PROCESSED_CONCEPTS_DATA_TEST_FILENAME_TEMPLATE
from utils.data import format_processed_filename
from utils.io import load_pickle, save_pickle


def load_data():
    filename = format_processed_filename(PROCESSED_DATA_DIR, PROCESSED_DATA_FILENAME_TEMPLATE, genre='clinical')
    data_train, data_dev = load_pickle(filename)

    print('Loaded:', filename.name)
    return data_train, data_dev


def load_data_test():
    filename = format_processed_filename(PROCESSED_DATA_DIR, PROCESSED_DATA_TEST_FILENAME_TEMPLATE, genre='clinical')
    data_test = load_pickle(filename)

    print('Loaded:', filename.name)
    return data_test


def save_data(data_train, data_dev):
    filename = format_processed_filename(PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE,
                                         genre='clinical')
    save_pickle(filename, (data_train, data_dev))

    print('Saved:', filename.name)


def save_data_test(data_test):
    filename = format_processed_filename(PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_TEST_FILENAME_TEMPLATE,
                                         genre='clinical')
    save_pickle(filename, data_test)

    print('Saved:', filename.name)


def extract_semantic_types(semtypes):
    return semtypes.replace('[', '').replace(']', '').split(',')


def extract_positional_information(pos_info):
    pos_info_parsed = pos_info.replace('[', '').replace(']', '').replace(';', ',').split(',')
    pos_info_parsed = [p.split('/') for p in pos_info_parsed]
    pos_info_parsed = [(int(p[0]) - 1, int(p[0]) - 1 + int(p[1])) for p in pos_info_parsed]

    return pos_info_parsed


def process_sentences(sentences):
    mm = MetaMap.get_instance(METAMAP_BINARY_PATH)

    sentences_ids = list(range(len(sentences)))
    concepts, error = mm.extract_concepts(sentences, sentences_ids)

    sentences_concepts = [[] for _ in range(len(sentences))]
    for concept in concepts:
        if not isinstance(concept, ConceptMMI):
            continue

        sentence_id = int(concept.index)

        concept_data = {
            'preferred_name': concept.preferred_name,
            'cui': concept.cui,
            'pos_info': extract_positional_information(concept.pos_info),
            'semtypes': extract_semantic_types(concept.semtypes),
            'score': float(concept.score),
        }
        sentences_concepts[sentence_id].append(concept_data)

    return sentences_concepts


def main():
    data_train, data_dev = load_data()
    print('Train:', [len(d) for d in data_train], 'dev:', [len(d) for d in data_dev])

    sentences_premise_train = data_train[0]
    sentences_hypothesis_train = data_train[1]
    sentences_premise_dev = data_dev[0]
    sentences_hypothesis_dev = data_dev[1]

    concepts_premise_train = process_sentences(sentences_premise_train)
    concepts_hypothesis_train = process_sentences(sentences_hypothesis_train)
    concepts_premise_dev = process_sentences(sentences_premise_dev)
    concepts_hypothesis_dev = process_sentences(sentences_hypothesis_dev)

    data_train = (data_train[0], data_train[1], data_train[2], concepts_premise_train, concepts_hypothesis_train)
    data_dev = (data_dev[0], data_dev[1], data_dev[2], concepts_premise_dev, concepts_hypothesis_dev)
    print('Concepts - train:', [len(d) for d in data_train], 'dev:', [len(d) for d in data_dev])

    save_data(data_train, data_dev)


def main_data_test():
    data_test = load_data_test()
    print('Test:', [len(d) for d in data_test])

    sentences_premise_test = data_test[0]
    sentences_hypothesis_test = data_test[1]

    concepts_premise_test = process_sentences(sentences_premise_test)
    concepts_hypothesis_test = process_sentences(sentences_hypothesis_test)

    data_test = (data_test[0], data_test[1], data_test[2], concepts_premise_test, concepts_hypothesis_test)
    print('Concepts - test:', [len(d) for d in data_test])

    save_data_test(data_test)


if __name__ == '__main__':
    main()
