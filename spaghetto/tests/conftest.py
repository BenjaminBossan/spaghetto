from functools import partial
import random

import numpy as np
import pytest

from lasagne.layers import EmbeddingLayer
from lasagne.layers import InputLayer
from lasagne.layers.recurrent import GRULayer
from lasagne.updates import rmsprop
from lasagne import nonlinearities

from spaghetto.model import RNN
from spaghetto.recurrent import RNNDenseLayer
from spaghetto.utils import TokenEncoder


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")

    def __enter__(self):
        for k, v in self.dct.items():
            self.attr_before[k] = getattr(self.obj, k)
            setattr(self.obj, k, v)
        return self.obj

    def __exit__(self, *args, **kwargs):
        for k, v in self.attr_before.items():
            setattr(self.obj, k, v)
        return self.obj


def mean_relative_improvement(losses):
    abs_improvement = losses[0] - losses[-1]
    rel_improvement = abs_improvement / losses[0]
    mean_rel_improvement = rel_improvement / len(losses)
    return mean_rel_improvement


def data_abc_linewise_X():
    X = ['aaaabc' for __ in range(1000)]
    return X


def data_abc_one_string_X():
    X = data_abc_linewise_X()
    return '\n'.join(X[0])


def rnn_kwargs():
    layers = [
        (InputLayer, {}),
        (EmbeddingLayer, {'output_size': 10}),
        (GRULayer, {'grad_clipping': 5., 'num_units': 20}),
        (GRULayer, {'grad_clipping': 5., 'num_units': 20}),
        (RNNDenseLayer, {'nonlinearity': nonlinearities.identity}),
    ]

    kwargs = {
        'layers': layers,
        'batch_size': 8,
        'max_epochs': 1,
        'verbose': 1,
        'max_len_line': 20,
        'patience': None,
        'updater': partial(rmsprop, learning_rate=1e-2),
    }
    return kwargs


class BaseTest:
    @pytest.fixture(scope='class')
    def encoder(self):
        encoder = TokenEncoder().fit(self.X)
        return encoder

    @pytest.fixture(scope='class')
    def clf(self, encoder):
        clf = RNN(encoder=encoder, **rnn_kwargs())
        clf.initialize()
        clf.fit(self.X, num_epochs=1)
        return clf

    @pytest.mark.slow
    def test_fit(self, clf, encoder):
        clf.fit(self.X)
        clf.fit(self.X, on_nth_batch=5, num_epochs=2, max_len_sample=5)

    @pytest.mark.slow
    def test_fit_without_valid(self, clf, encoder):
        old_eval_size = clf.eval_size
        clf.eval_size = 0.
        self.test_fit(clf, encoder)
        clf.eval_size = old_eval_size

    @pytest.mark.slow
    def test_fit_witout_verbose(self, clf):
        old_verbose = clf.verbose
        clf.verbose = 0

        clf.fit(self.X, num_epochs=1)

        clf.verbose = old_verbose

    @pytest.mark.slow
    def test_predict_proba(self, clf):
        probs = clf.predict_proba(self.X[:75])
        assert probs.ndim == 3
        assert probs.dtype == np.float32

        clf.predict_proba(self.X[:40], 44)

    @pytest.mark.slow
    def test_predict(self, clf):
        pred = clf.predict(self.X[:100])
        assert isinstance(pred[0], (str, unicode))

        clf.predict(self.X[:50], 1.7, 7)

    @pytest.mark.slow
    def test_sample(self, clf):
        sample = clf.sample(1)
        assert isinstance(sample[0], (str, unicode))

        clf.sample(3, temperature=0.1, max_len_line=23)
