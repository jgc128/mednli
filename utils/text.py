import logging

import itertools
import re
from collections import defaultdict

from nltk.stem.porter import PorterStemmer


def find_all(a_str, sub):
    """
    https://stackoverflow.com/a/4665027
    """
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def is_time(token):
    without_time = re.sub(r'(\d)*(\d):(\d\d)([aA][mM]|[pP][Mm])', '', token).strip()
    return not without_time


def is_num(token):
    try:
        num = float(token)
        return True
    except ValueError as e:
        return False


def is_non_alpha(token):
    without_alpha = re.sub(r'[^A-Za-z]', '', token).strip()
    return not without_alpha


def remove_punctuation(token):
    without_punctuation = re.sub(r'[^A-Za-z0-9]', '', token)
    return without_punctuation


def clean_sentence(sentence):
    tokens = sentence.split(' ')

    sentence_cleaned_tokens = []
    for i, token in enumerate(tokens):
        if is_time(token):
            sentence_cleaned_tokens.append('TIME')
            continue

        if is_num(token):
            sentence_cleaned_tokens.append('NUM')
            continue

        if is_non_alpha(token):
            continue

        token = remove_punctuation(token)
        sentence_cleaned_tokens.append(token)

    sentence_cleaned = ' '.join(sentence_cleaned_tokens)

    return sentence_cleaned


def clean_list_of_sentences(sentences):
    sentences_cleaned = [clean_sentence(s) for s in sentences]
    return sentences_cleaned


def clean_data(data):
    if data is None:
        return None

    premise_cleaned = clean_list_of_sentences(data['premise'])
    hypothesis_cleaned = clean_list_of_sentences(data['hypothesis'])

    data['premise'] = premise_cleaned
    data['hypothesis'] = hypothesis_cleaned

    return data


def stem_list_of_sentences(sentences, stemmer):
    sentences = [s.split() for s in sentences]
    sentences = [[stemmer.stem(t) for t in s] for s in sentences]
    sentences = [' '.join(s) for s in sentences]

    return sentences


def stem_data(data):
    if data is None:
        return None

    stemmer = PorterStemmer()

    premise_stemmed = stem_list_of_sentences(data['premise'], stemmer)
    hypothesis_stemmed = stem_list_of_sentences(data['hypothesis'], stemmer)

    data['premise'] = premise_stemmed
    data['hypothesis'] = hypothesis_stemmed

    return data


def replace_str_to_list(sentence, replacements):
    """For each (new_str, position) in the list replace characters in the position to new_str"""
    # filter out replacement at the same location
    replacements_uniq = {}
    for to_str, pos in replacements:
        if pos not in replacements_uniq:
            replacements_uniq[pos] = to_str

    # sort by desc
    replacements_pos = sorted(replacements_uniq.keys(), key=lambda x: x[0], reverse=True)

    for pos in replacements_pos:
        to_str = replacements_uniq[pos]
        sentence = sentence[:pos[0]] + to_str + sentence[pos[1]:]

    return sentence


def replace_cui_sentences(sentences, cuis, target_cuis):
    sentences_replaced = []

    number_of_replacements_before = 0
    number_of_replacements_after = 0

    for sentence, sentence_cuis in zip(sentences, cuis):
        replacements = [
            (cui['cui'], postition, cui['score'])
            for cui in sentence_cuis for postition in cui['pos_info']
        ]
        replacements = [r for r in replacements if r[0] in target_cuis]
        number_of_replacements_before += len(replacements)

        # if there are several candidates - select one with the higher score
        replacements_uniq = {}
        replacements_uniq_scores = defaultdict(lambda: 0)
        for cui, position, score in replacements:
            if replacements_uniq_scores[position] < score:
                replacements_uniq[position] = cui

        replacements = [(cui, position) for position, cui in replacements_uniq.items()]
        number_of_replacements_after += len(replacements)

        sentence_replaced = replace_str_to_list(sentence, replacements)
        sentences_replaced.append(sentence_replaced)

    logging.info('Replaced CUIs: %s -> %s', number_of_replacements_before, number_of_replacements_after)

    return sentences_replaced


def replace_cui_data(data, target_cuis):
    if data is None:
        return None

    premise_replaced = replace_cui_sentences(data['premise'], data['premise_concepts'], target_cuis)
    hypothesis_replaced = replace_cui_sentences(data['hypothesis'], data['hypothesis_concepts'], target_cuis)

    data['premise'] = premise_replaced
    data['hypothesis'] = hypothesis_replaced

    return data


def lowecase_list_of_sentences(sentences):
    sentences_lowercased = [s.lower() for s in sentences]
    return sentences_lowercased


def lowercase_data(data):
    if data is None:
        return None

    premise_lowercased = lowecase_list_of_sentences(data['premise'])
    hypothesis_lowercased = lowecase_list_of_sentences(data['hypothesis'])

    data['premise'] = premise_lowercased
    data['hypothesis'] = hypothesis_lowercased

    return data


def get_tokens_cuis(sentence, sentence_cuis, delimiter=' '):
    tokens_boundaries = []

    token_start = 0
    for i, token_end in enumerate(find_all(sentence, delimiter)):
        tokens_boundaries.append((token_start, token_end))

        token_start = token_end + 1

    tokens_boundaries.append((token_start, len(sentence)))

    tokens_cuis = defaultdict(list)

    for token_id, (token_start, token_end) in enumerate(tokens_boundaries):
        for cui_info in sentence_cuis:
            for cui_pos in cui_info['pos_info']:

                if cui_pos[0] <= token_start and cui_pos[1] >= token_end:
                    tokens_cuis[token_id].append(cui_info)

    return tokens_cuis
