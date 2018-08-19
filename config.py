from pathlib import Path

DATA_DIR = Path('./data/')

try:
    from local_config import *
except ImportError:
    pass

SNLI_DIR = DATA_DIR / 'snli_1.0/'
SNLI_TRAIN_FILENAME = SNLI_DIR / 'snli_1.0_train.jsonl'
SNLI_DEV_FILENAME = SNLI_DIR / 'snli_1.0_dev.jsonl'
SNLI_TEST_FILENAME = SNLI_DIR / 'snli_1.0_dev.jsonl'

MULTINLI_DIR = DATA_DIR / 'multinli_0.9/'
MULTINLI_TRAIN_FILENAME = MULTINLI_DIR / 'multinli_0.9_train.jsonl'
MULTINLI_DEV_FILENAME = MULTINLI_DIR / 'multinli_0.9_dev_matched.jsonl'

MLI_DIR = DATA_DIR / 'mli_2.0/'
MLI_TRAIN_FILENAME = MLI_DIR / 'mli_train_v2.jsonl'
MLI_DEV_FILENAME = MLI_DIR / 'mli_dev_v2.jsonl'
MLI_TEST_FILENAME = MLI_DIR / 'mli_test_v2.jsonl'

WORD_VECTORS_DIR = DATA_DIR / 'word_embeddings/'
WORD_VECTORS_FILENAME = {
    'glove': WORD_VECTORS_DIR / 'glove.840B.300d.pickled',
    'mimic': WORD_VECTORS_DIR / 'mimic.fastText.no_clean.300d.pickled',
    'bio_asq': WORD_VECTORS_DIR / 'bio_asq.no_clean.300d.pickled',
    'wiki_en': WORD_VECTORS_DIR / 'wiki_en.fastText.300d.pickled',
    'wiki_en_mimic': WORD_VECTORS_DIR / 'wiki_en_mimic.fastText.no_clean.300d.pickled',
    'glove_bio_asq': WORD_VECTORS_DIR / 'glove_bio_asq.no_clean.300d.pickled',
    'glove_bio_asq_mimic': WORD_VECTORS_DIR / 'glove_bio_asq_mimic.no_clean.300d.pickled',
    'crawl': WORD_VECTORS_DIR / 'crawl.fastText.300d.pkl',

    'glove_synonyms': WORD_VECTORS_DIR / 'glove.840B.300d.wordnet-synonyms.pickled',
    'glove_synonyms+': WORD_VECTORS_DIR / 'glove.840B.300d.wordnet-synonyms+.pickled',
}

PROCESSED_DATA_DIR = DATA_DIR / 'nli_processed/'
PROCESSED_DATA_FILENAME_TEMPLATE = 'genre_{genre}.pkl'
PROCESSED_DATA_TEST_FILENAME_TEMPLATE = 'genre_{genre}_test.pkl'
PROCESSED_CONCEPTS_DATA_FILENAME_TEMPLATE = 'genre_concepts_{genre}.pkl'
PROCESSED_CONCEPTS_DATA_TEST_FILENAME_TEMPLATE = 'genre_concepts_{genre}_test.pkl'

MODELS_WEIGHTS_DIR = DATA_DIR / 'saved_models/'

METAMAP_BINARY_PATH = DATA_DIR / 'umls/metamap/public_mm/bin/metamap'
UMLS_INSTALLATION_DIR = DATA_DIR / 'umls/installation/2017AA'
UMLS_CONCEPTS_GRAPH_FILENAME = DATA_DIR / 'umls_concepts_graph.pickled'

RETROFITTING_DIR = DATA_DIR / 'retrofitting'
