import math
import random
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tensorflow.keras.utils import Sequence


def fix_columns(s):
    return s.replace("|", "").strip()


def extract_X(s):
    if len(s) > 0:
        x = [int(i) + 3 for i in s.split(" ")]
        return [1] + x + [2]
    else:
        return []


def extract_y(s, length=17):
    y = [0] * length
    if len(s) > 0:
        for i in s.split(" "):
            y[int(i)] = 1
    return y


def export_result(file, testy):
    result = pd.DataFrame(testy).applymap(
        lambda x: f'{x:0.10f}').apply(lambda x: " ".join(x), axis=1)
    result = result.reset_index()
    result.columns = ["report_ID", "Prediction"]
    with open(file, "w") as f:
        f.writelines(result.apply(
            lambda x: f"{x[0]}|,|{x[1]}\n", axis=1).tolist())


def k_fold(X, *others, n_splits=5, random_state=2021):
    data_size = len(X)
    for o in others:
        assert len(o) == data_size

    np.random.seed(random_state)
    indexes = np.random.permutation(data_size)

    split_size = data_size // n_splits

    for i in range(n_splits):
        test_indexes = indexes[i*split_size: (i+1)*split_size]
        train_indexes = np.setdiff1d(indexes, test_indexes)
        yield [(np.array(x)[train_indexes], np.array(x)[test_indexes]) for x in chain([X], others)]


class TransProbaTransformer(BaseEstimator, TransformerMixin):
    # 仅适用于本次比赛
    def __init__(self):
        self._init()

    def _init(self):
        self.pair_counter = Counter()
        self.start_counter = Counter()
        self.trains_dict = dict()

    def fit(self, X):
        self._init()
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        for x in chain(*X.str[:-1]):
            self.start_counter[x] += 1

        for x_pair in chain(*X.apply(lambda x: zip(x[:-1], x[1:]))):
            self.pair_counter[tuple(x_pair)] += 1

        for p, c in self.pair_counter.items():
            self.trains_dict[p] = c / self.start_counter[p[0]]

        return self

    def get_transproba(self, pairs):
        return [0] + [self.trains_dict.get(tuple(p), 0) for p in pairs]

    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        trans_proba = X.apply(lambda x: zip(
            x[:-1], x[1:])).apply(self.get_transproba)
        return trans_proba


class ImageDesSequence(Sequence):
    def __init__(self, X, y, batch_size, train=True, shuffle=True, musk=0, y_not_null=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle
        if musk != 0:
            self.X = self.musk(self.X, musk)
        if y_not_null:
            self.X = self.X[self.y.sum(axis=1) != 0]
            self.y = self.y[self.y.sum(axis=1) != 0]

        if shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        idxes = np.random.permutation(len(self.X))
        self.X = self.X[idxes]
        self.y = self.y[idxes]

    def musk(self, x, musk_rate=0.1):
        size_ = x.shape
        return x * np.random.choice([True, False], size_, p=[1-musk_rate, musk_rate])

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        x = self.X[begin:end]
        if self.train:
            return x, self.y[begin:end]
        else:
            return x

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()


class MultiLabelOverSampler:
    def __init__(self, random_state=0):
        np.random.seed(random_state)

    def fit_resample(self, X, y, shuffule_at_last=True):
        finnal_indexes = list(range(len(y)))
        init_count = y.sum(axis=0)
        target_size = init_count.max()

        for i in range(len(init_count)):
            label_count = y[finnal_indexes].sum(axis=0)
            current_size = label_count.min()
            to_fill_size = target_size - current_size
            already_filled_labels = np.where(label_count >= target_size)[0]
            to_fill_num = np.where(label_count == current_size)[0][0]
            if len(already_filled_labels) == len(label_count):
                break

            satisfy_indexes = np.where((y[:, already_filled_labels] == 0).all(
                axis=1) & (y[:, to_fill_num] == 1))[0]
            add_indexes = np.random.choice(satisfy_indexes, to_fill_size)
            finnal_indexes += add_indexes.tolist()

        if shuffule_at_last:
            np.random.shuffle(finnal_indexes)

        return X[finnal_indexes], y[finnal_indexes]
