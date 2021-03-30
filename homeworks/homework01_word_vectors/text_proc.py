#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer


def get_bow_keys(texts_train, max_size=10000):
    occurances = ' '.join(texts_train).split()
    keys = set(occurances)
    k = min(max_size, len(keys))
    return keys, occurances, k


def get_bow_rating(occurances, keys):
    bow_rating = dict.fromkeys(keys, 0)  # <YOUR CODE>
    for w in occurances:
        bow_rating[w] += 1
    return bow_rating


def get_bow_vocabulary(texts, max_size=10000):
    keys, occurances, k = get_bow_keys(texts, max_size)

    bow_rating = get_bow_rating(occurances=occurances, keys=keys)

    highly_rated = sorted(bow_rating.items(), key=lambda item: item[1], reverse=True)[:k]
    return [k for k, v in highly_rated]


def text_to_bow_ex(text, bow_vocabulary):
    """ convert text string to an array of token counts. Use bow_vocabulary. """
    bow = dict.fromkeys(bow_vocabulary, 0)
    for wrd in text.split():
        if wrd in bow_vocabulary:
            bow[wrd] += 1

    return np.array(list(bow.values()), 'float32')


def text_to_bow(text):
    return text_to_bow_ex(text, bow_vocabulary)


# ===============================================================


def get_tf(text):
    tf = {}
    for wrd in text.split():
        if wrd in tf:
            tf[wrd] += 1
        else:
            tf[wrd] = 1
    for wk in tf:
        tf[wk] /= len(text)
    return tf


def get_idf(tf):
    count = {}
    sum = 0
    for txt in tf:
        for wrd in txt:
            if wrd in count:
                count[wrd] += 1
            else:
                count[wrd] = 1
        sum += len(txt)
    idf = {}
    for wrd in count:
        idf[wrd] = np.log(sum / (count[wrd] + alpha))
    return idf, count


def get_tfidf(tf, idf):
    tfidf = {}
    for wrd in tf:
        tfidf[wrd] = tf[wrd] * idf[wrd]
    return tfidf


def get_tfidf_vector(text, tfidf, count):
    word_dict = sorted(count.keys())
    tfidf_vector = [0.0] * len(word_dict)
    for i, wrd in enumerate(word_dict):
        if wrd in text:
            tfidf_vector[i] = text[wrd]
    return tfidf_vector


if __name__ == '__main__':
    data = pd.read_csv("comments.tsv", sep='\t')
    texts = data['comment_text'].values
    target = data['should_ban'].values
    texts_train, texts_test, y_train, y_test = train_test_split(texts, target, test_size=0.5, random_state=42)
    tokenizer = TweetTokenizer()
    preprocess = lambda text: ' '.join(tokenizer.tokenize(text.lower()))
    texts_train = np.array(list(map(preprocess, texts_train)))
    texts_test = np.array(list(map(preprocess, texts_test)))

    bow_vocabulary = get_bow_vocabulary(texts_train, 10000)

    keys, occurances, k = get_bow_keys(texts_train, max_size=10000)
    bow_rating = get_bow_rating(occurances=occurances, keys=keys)

    alpha = 1

    tf_train = list(map(get_tf, texts_train))
    tf_test = list(map(get_tf, texts_test))
    # print(tf_train[0])

    idf_train, count_train = get_idf(tf_train)
    idf_test, count_test = get_idf(tf_test)
    # print(idf_train)

    tfidf_train = [get_tfidf(t, idf_train) for t in tf_train]
    tfidf_test = [get_tfidf(t, idf_test) for t in tf_test]

    tfidf_vector_train = [get_tfidf_vector(txt, tfidf_train, count_train) for txt in tfidf_train]
    tfidf_vector_test = [get_tfidf_vector(txt, tfidf_test, count_test) for txt in tfidf_test]

