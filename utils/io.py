import csv
import pickle
import logging

from config import WORD_VECTORS_FILENAME, WORD_VECTORS_DIR, RETROFITTING_DIR


def load_pickle(filename):
    try:
        with open(str(filename), 'rb') as f:
            obj = pickle.load(f)

        logging.info('Loaded: %s', filename)

    except EOFError:
        logging.warning('Cannot load: %s', filename)
        obj = None

    return obj


def save_pickle(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)

    logging.info('Saved: %s', filename)


def save_csv(filename, data, fieldnames=None, flush=False):
    with open(str(filename), 'w') as f:
        if fieldnames is not None:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        else:
            writer = csv.writer(f)

        if not flush:
            writer.writerows(data)
        else:
            # write line by line and flush after each line
            for row in data:
                writer.writerow(row)
                f.flush()


def get_word_vectors_filename(name):
    if name in WORD_VECTORS_FILENAME:
        filename = WORD_VECTORS_FILENAME[name]
    else:
        filename = WORD_VECTORS_DIR / name
        if not filename.exists():
            filename = RETROFITTING_DIR / name

    return filename
