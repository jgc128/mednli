from collections import Counter

import itertools

from config import PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE
from utils.data import format_processed_filename, load_single_genre_data
from utils.io import load_pickle, save_pickle


def find_uniq_concepts():
    data_train, data_dev = load_single_genre_data(
        PROCESSED_DATA_DIR, PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE, genre='clinical')
    print('Train:', [len(d) for d in data_train], 'dev:', [len(d) for d in data_dev])

    concepts = itertools.chain(data_train[3], data_train[4], data_dev[3], data_dev[4])
    concepts = itertools.chain.from_iterable(concepts)

    concepts_counter = Counter()
    for i, concept in enumerate(concepts):
        cui = concept['cui']
        concepts_counter[cui] += 1

    return concepts_counter


def main():
    concepts_counter = find_uniq_concepts()

    print(concepts_counter.most_common(10))
    print('Uniq concepts:', len(concepts_counter))

    for concept, count in concepts_counter.most_common():
        print(concept)


if __name__ == '__main__':
    main()
