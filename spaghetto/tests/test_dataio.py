import numpy as np
import pytest

from spaghetto.dataio import batch_streamer
from spaghetto.dataio import valid_batch_size


@pytest.mark.parametrize('bs, es', [
    (5, 0),
    (100, 0),
    (5, 0.1),
    (10, 0.1),
    (111, 0.1),
    (103, 0.2),
    (43, 0.5),
    (10, 0.9),
])
def test_valid_batch_size(bs, es):
    # bs: batch size train
    # es: eval size valid
    bs_valid = valid_batch_size(bs, es)
    bs_expected = np.round((bs_valid + bs) * es)
    assert bs_expected == bs_valid


def test_batch_streamer_raises():
    with pytest.raises(TypeError):
        X = 123  # not string, unicode or list
        batch_streamer(
            X, 10, 0.3, {'a': 0, 'b': 1}, lambda x: x, 3, 'a', 'b').next()
