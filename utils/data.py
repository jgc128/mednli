import logging
import numpy as np

from utils.io import load_pickle


def format_processed_filename(dir, filename_template, **kwargs):
    """Obtain the filename of the processed data base on the provided template and parameters"""
    filename = dir / filename_template.format(**kwargs)
    return filename


def load_single_genre_data(directory, filename_template, genre, filename_test_template=None):
    lambda_concepts = lambda d: {
        'premise': d[0],
        'hypothesis': d[1],
        'label': d[2],
        'premise_concepts': d[3],
        'hypothesis_concepts': d[4],
    }
    lambda_no_concepts = lambda d: {
        'premise': d[0],
        'hypothesis': d[1],
        'label': d[2],
        'premise_concepts': [],
        'hypothesis_concepts': [],
    }

    filename = format_processed_filename(directory, filename_template, genre=genre)
    data_train, data_dev = load_pickle(filename)

    if len(data_train) > 3:
        tuple_to_dict = lambda_concepts
    else:
        tuple_to_dict = lambda_no_concepts

    data_train = tuple_to_dict(data_train)
    data_dev = tuple_to_dict(data_dev)

    data_test = None
    if filename_test_template is not None:
        filename_test = format_processed_filename(directory, filename_test_template, genre=genre)
        data_test = load_pickle(filename_test)

        if len(data_test) <= 3:
            tuple_to_dict = lambda_no_concepts

        data_test = tuple_to_dict(data_test)

    if filename_test_template is None:
        return data_train, data_dev
    else:
        return data_train, data_dev, data_test


def load_processed_genre_data(directory, filename_template, genre_source, genre_target=None, genre_tune=None):
    """Load the processed data for different source/target/tune genres"""
    data_source_train, data_source_dev = load_single_genre_data(directory, filename_template, genre_source)

    data_target_train = None
    data_target_dev = None
    if genre_target is not None and genre_target != '':
        data_target_train, data_target_dev = load_single_genre_data(directory, filename_template, genre_target)

    data_tune_train = None
    data_tune_dev = None
    if genre_tune is not None and genre_tune != '':
        data_tune_train, data_tune_dev = load_single_genre_data(directory, filename_template, genre_tune)

    return data_source_train, data_source_dev, data_target_train, data_target_dev, data_tune_train, data_tune_dev
