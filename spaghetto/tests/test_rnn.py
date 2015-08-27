# -*- coding: utf-8 -*-

from __future__ import division
import sys

from mock import Mock
import numpy as np
import pytest

from .conftest import BaseTest
from .conftest import data_abc_one_string_X
from .conftest import data_abc_linewise_X
from spaghetto.model import RNN


sys.setrecursionlimit(5000)


class TestRNNFunctionsOnlyOnce:
    def test_init_raises_when_patience_and_no_eval(self):
        with pytest.raises(ValueError):
            RNN(layers=[], encoder=Mock(), patience=3, eval_size=0)


class TestRNNSaveLoadPatience(BaseTest):
    X = data_abc_linewise_X()

    @pytest.fixture
    def mock_clf(self, clf, monkeypatch):
        monkeypatch.setattr(
            clf, 'valid_iter_', Mock(side_effect=[1 for __ in range(10000)]))
        # train_iter_ does not need to be mocked, but it is faster this way
        monkeypatch.setattr(
            clf, 'train_iter_', Mock(side_effect=[1 for __ in range(10000)]))
        return clf

    @pytest.mark.slow
    def test_save_and_load(self, clf):
        p_init = clf.predict_proba(self.X[0])
        clf.save_params_to('tmp')

        clf.fit(self.X, num_epochs=1)
        p_inter = clf.predict_proba(self.X[0])

        assert not (p_init == p_inter).all()

        clf.load_params_from('tmp')
        p_end = clf.predict_proba(self.X[0])

        assert (p_init == p_end).all()

    @pytest.mark.parametrize('patience', [1, 3, 7])
    def test_patience_constant_loss(self, mock_clf, patience, monkeypatch):
        monkeypatch.setattr(mock_clf, 'patience', patience)

        epochs_before = mock_clf.train_history_[-1]['epoch']
        mock_clf.fit(self.X, num_epochs=10)
        epochs_after = mock_clf.train_history_[-1]['epoch']
        assert epochs_after - epochs_before - 1 == patience

    @pytest.mark.parametrize('patience', [1, 4, 6])
    def test_patience_decreasing_loss_improvement_threshold_1(
            self, mock_clf, patience, monkeypatch):
        monkeypatch.setattr(mock_clf, 'patience', patience)
        monkeypatch.setattr(mock_clf, 'improvement_threshold', 1)
        monkeypatch.setattr(
            mock_clf, 'valid_iter_', Mock(side_effect=np.linspace(2, 1, 1000)))

        epochs_before = mock_clf.train_history_[-1]['epoch']
        mock_clf.fit(self.X, num_epochs=10)
        epochs_after = mock_clf.train_history_[-1]['epoch']
        assert epochs_after - epochs_before == 10

    @pytest.mark.parametrize('patience', [1, 4, 6])
    def test_patience_decreasing_loss_improvement_threshold_lt_improvement(
            self, mock_clf, patience, monkeypatch):
        monkeypatch.setattr(mock_clf, 'patience', patience)
        monkeypatch.setattr(mock_clf, 'improvement_threshold', 0.9)
        monkeypatch.setattr(
            mock_clf, 'valid_iter_', Mock(side_effect=np.linspace(2, 1, 1000)))

        epochs_before = mock_clf.train_history_[-1]['epoch']
        mock_clf.fit(self.X, num_epochs=10)
        epochs_after = mock_clf.train_history_[-1]['epoch']
        assert epochs_after - epochs_before - 1 == patience


class TestRNNLinewiseX(BaseTest):
    X = data_abc_linewise_X()


class TestRNNOneString(BaseTest):
    X = data_abc_one_string_X()
