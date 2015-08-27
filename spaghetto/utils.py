# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import OrderedDict
import re

import numpy as np
from tabulate import tabulate
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams


floatX = theano.config.floatX
srng = MRG_RandomStreams(np.random.randint(1, 17411))
BLUE = '\033[94m'
CYAN = '\033[36m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
RED = '\033[31m'
ENDC = '\033[0m'


def init_random(shape0, shape1, scale=0.16, distr='uniform'):
    if distr == 'normal':
        arr = scale * np.random.randn(shape0, shape1).astype(floatX)
    elif distr == 'uniform':
        arr = scale * np.random.randn(shape0, shape1) - 0.5 * scale
        arr = arr.astype(floatX)
    return theano.shared(arr)


def init_zeros(*shape):
    return theano.shared(np.zeros(shape, dtype=floatX))


class TokenEncoder(object):
    def __init__(
        self,
        separator=None,
        start_token='<START>',
        end_token='\n',
        special_tokens=[],
    ):
        """TokenEncoder class for use in conjunction with spaghetto.RNN.

        Parameters
        ----------
        separator : ' ' or None (default=None)
          How to separate tokens. The default, None, should be used
          for character-level tokenization. Use space (' ') for word
          level tokenization. Other techniques are not currently
          supported.

        start_token : str (default='<START>')
          This token signals to the RNN that a sequence begins. It
          should be a token that never occurs in the training data.

        end_token : str (default='\n')
          This token signals the end of a sequence and defaults to the
          line break. There might be occasions where a line break
          should not signal the end of a token.

        special_tokens : list of strings (default=[])
          If you have special tokens that you don't want the encoder
          to break, list them here.

        """
        if not ((separator is None) or (separator == ' ')):
            raise NotImplementedError("Separator must be None or ' ' (space)"
                                      "at the moment.")

        self.separator = separator
        self.start_token = start_token
        self.end_token = end_token
        self.special_tokens = special_tokens

    def fit(self, X):
        if isinstance(self.special_tokens, (str, unicode)):
            special_tokens = [self.special_tokens]
        else:
            special_tokens = self.special_tokens
        special_tokens += [self.start_token, self.end_token]

        # prepare splits
        if self.separator == " ":
            splits = re.compile('\s+').split
        elif self.separator is None:
            splits = re.compile('|'.join(special_tokens + ['.'])).findall
        self.splits_ = splits

        # prepare join method
        def join(tokens):
            if self.separator is None:
                indices = "".join(tokens)
            else:
                indices = self.separator.join(tokens)
            return indices
        self.join = join

        # prepare set of tokens
        tokenset = set()
        if isinstance(X, str):
            tokenset.update(self.splits_(X))
        else:
            for line in X:
                tokenset.update(self.splits_(line))

        if self.start_token in tokenset:
            raise ValueError(
                "Start token {} is part of the training data"
                "".format(self.start_token))

        # add end_token, and start_token to tokenset
        tokenset.update(set(special_tokens))
        tokenset = sorted(tokenset)
        self.tokenset_ = tokenset
        self.num_tokens_ = len(tokenset)

        # translation from token to index and vice versa
        self.id2token_ = list(tokenset)
        self.token2id_ = {token: idx for idx, token in
                          enumerate(self.id2token_)}

        return self

    def transform(self, X):
        Xt = []
        for x in X:
            tokens = self.splits_(x.strip(' '))
            indices = [self.token2id_[token] for token in tokens]
            Xt.append(indices)
        return Xt

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Xt):
        X = []
        for indices in Xt:
            tokens = [self.id2token_[idx] for idx in indices]
            x = self.join(tokens)
            X.append(x)
        return X


class PrintLog:  # pragma: no cover
    def __init__(self):
        self.first_iteration = True
        self.best_valid = np.inf
        self.best_train = np.inf

    def __call__(self, train_history):
        print(self.table(train_history))

    def table(self, train_history):
        best_valid = self.best_valid
        best_train = self.best_train
        info = train_history[-1]
        max_len_line = len(info['sample'])
        sample_str = u"".join(["{:<", str(max_len_line), "}"])

        valid_loss = info['valid_loss']
        if valid_loss < best_valid:
            self.best_valid = valid_loss
            valid_loss = "{}{:.4f}{}".format(GREEN, valid_loss, ENDC)
        else:
            valid_loss = "{:.4f}".format(valid_loss)

        train_loss = info['train_loss']
        if train_loss < best_train:
            self.best_train = train_loss
            train_loss = "{}{:4f}{}".format(CYAN, train_loss, ENDC)
        else:
            train_loss = "{:.4f}".format(train_loss)

        t_div_v = info['train_loss'] / info['valid_loss']
        train_valid = "{:4f}".format(t_div_v if np.isfinite(t_div_v) else 0)

        duration = "{:.2f}s".format(info['duration'])

        sample = sample_str.format(info['sample'])

        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train perpl.', train_loss),
            ('valid perpl.', valid_loss),
            ('train/valid', train_valid),
            ('duration', duration),
            (sample_str.format('sample'), sample),
        ])

        tabulated = tabulate(
            [info_tabulate], headers='keys', floatfmt='.5f')

        out = ""
        if self.first_iteration:
            out = "\n"
            out += "\n".join(tabulated.split("\n", 2)[:2])
            out += "\n"
            self.first_iteration = False

        out += tabulated.rsplit("\n", 1)[-1]
        return out
