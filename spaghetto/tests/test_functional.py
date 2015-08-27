from functools import partial

import pytest

from lasagne.layers import InputLayer
from lasagne.layers import EmbeddingLayer
from lasagne.nonlinearities import identity
from lasagne.layers.recurrent import GRULayer
from lasagne.layers.recurrent import LSTMLayer
from lasagne.updates import rmsprop

from spaghetto.model import RNN
from spaghetto.recurrent import RNNDenseLayer
from spaghetto.utils import TokenEncoder


@pytest.fixture(scope='session')
def X():
    X = ['1a1b1c1d' for __ in range(1000)]
    return X


@pytest.fixture(scope='session')
def encoder(X):
    encoder = TokenEncoder().fit(X)
    return encoder


@pytest.fixture(scope='session')
def gru(X, encoder):
    layers = [
        (InputLayer, {}),
        (EmbeddingLayer, {'output_size': 32}),
        (GRULayer, {'grad_clipping': 5., 'num_units': 100}),
        (GRULayer, {'grad_clipping': 5., 'num_units': 100}),
        (RNNDenseLayer, {'nonlinearity': identity}),
    ]
    rnn = RNN(
        layers=layers,
        encoder=encoder,
        verbose=1,
        improvement_threshold=1.,
        updater=partial(rmsprop, learning_rate=1e-2),
    )
    rnn.initialize()
    return rnn


@pytest.mark.slow
def test_gru_learns_simple_pattern(gru, X):
    gru.fit(X, on_nth_batch=5, num_epochs=3, max_len_sample=20)

    final_train_score = gru.train_history_[-1]['train_loss']
    final_valid_score = gru.train_history_[-1]['valid_loss']
    assert 1 < final_train_score <= 1.1
    assert 1 < final_valid_score <= 1.1

    preds = gru.sample(20)
    correct = [pred.startswith('1a1b1c1d') for pred in preds]
    assert sum(correct) >= 18


@pytest.fixture(scope='session')
def lstm(X, encoder):
    layers = [
        (InputLayer, {}),
        (EmbeddingLayer, {'output_size': 32}),
        (LSTMLayer, {'grad_clipping': 5., 'num_units': 100}),
        (LSTMLayer, {'grad_clipping': 5., 'num_units': 100}),
        (RNNDenseLayer, {'nonlinearity': identity}),
    ]
    rnn = RNN(
        layers=layers,
        encoder=encoder,
        verbose=1,
        improvement_threshold=1.,
        updater=partial(rmsprop, learning_rate=1e-2),
    )
    rnn.initialize()
    return rnn


@pytest.mark.slow
def test_lstm_learns_simple_pattern(lstm, X):
    lstm.fit(X, on_nth_batch=5, num_epochs=3, max_len_sample=20)

    final_train_score = lstm.train_history_[-1]['train_loss']
    final_valid_score = lstm.train_history_[-1]['valid_loss']
    assert 1 < final_train_score <= 1.1
    assert 1 < final_valid_score <= 1.1

    preds = lstm.sample(20)
    correct = [pred.startswith('1a1b1c1d') for pred in preds]
    assert sum(correct) >= 18
