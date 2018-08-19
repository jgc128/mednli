import json
import math

import pandas as pd
import numpy as np

LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
GENRES = ['fiction', 'government', 'slate', 'telephone', 'travel', 'snli', 'clinical']

ID_TO_LABEL = {i: l for l, i in LABELS.items()}


def read_nli_data(filename, set_genre=None, limit=None):
    """
    Read NLI data and return a DataFrame

    Optionally, set the genre column to `set_genre`
    """

    if limit is None:
        limit = float('inf')  # we could use math.inf in Python 3.5 :'(

    all_rows = []
    with open(str(filename)) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row['gold_label'] != '-':
                all_rows.append(row)

            if i > limit:
                break

    nli_data = pd.DataFrame(all_rows)

    if set_genre is not None:
        nli_data['genre'] = set_genre

    return nli_data


def get_tokens_from_parse(parse):
    """Parse a string in the binary tree SNLI format and return a string of joined by space tokens"""
    cleaned = parse \
        .replace('(', ' ').replace(')', ' ') \
        .replace('-LRB-', '(').replace('-RRB-', ')') \
        .replace('-LSB-', '[').replace('-RSB-', ']')

    tokens = cleaned.split()

    cleaned_string = ' '.join(tokens)

    # remove all non-ASCII characters for MetaMap
    cleaned_string = cleaned_string.encode('ascii', errors='ignore').decode()

    return cleaned_string
