import string

import numpy as np
import pytest

from spaghetto.utils import init_random
from spaghetto.utils import init_zeros
from spaghetto.utils import TokenEncoder

ALPHABET = string.ascii_lowercase


class TestInitRandom:
    @pytest.mark.parametrize('shape', [(1, 1), (2, 3), (4, 1)])
    def test_init_random_shape(self, shape):
        arr = init_random(shape[0], shape[1])
        resulting_shape = arr.get_value().shape
        assert shape == resulting_shape

    @pytest.mark.parametrize('scale', [0.1, 0.7, 123.])
    def test_init_random_scale(self, scale):
        arr = init_random(1000, 1000, scale)
        resulting_scale = np.std(arr.get_value())
        assert np.isclose(scale, resulting_scale, 0.01)

    @pytest.mark.parametrize('scale', [0.1, 0.7, 123.])
    def test_init_random_scale_normal_distribution(self, scale):
        arr = init_random(1000, 1000, scale, distr='normal')
        resulting_scale = np.std(arr.get_value())
        assert np.isclose(scale, resulting_scale, 0.01)


@pytest.mark.parametrize('shape', [(3,), (1, 4), (2, 3, 10)])
def test_init_zeros(shape):
    arr = init_zeros(*shape)
    resulting_shape = arr.get_value().shape
    assert shape == resulting_shape


class TestTokenizer:
    def test_tokenizer_creates_indices(self):
        X = 'abcdef'
        tokenizer = TokenEncoder().fit(X)

        # num tokens + start token + end token
        assert len(tokenizer.token2id_) == 8

        expected_keys = set(
            list('abcdef') + [tokenizer.start_token, tokenizer.end_token])
        assert set(tokenizer.token2id_.keys()) == expected_keys

    def test_tokenizer_encoded_and_decodes_simple_text(self):
        X = 'abcdef'
        tokenizer = TokenEncoder().fit(X)

        encoded = tokenizer.transform(['fedcba'])[0]
        assert len(encoded) == 6
        assert all([type(idx) == int for idx in encoded])

        decoded = tokenizer.inverse_transform([encoded])[0]
        assert decoded == 'fedcba'

    @pytest.mark.parametrize('line, expected', [
        ('', []),
        ('abc', ['a', 'b', 'c']),
        ('das ist', ['d', 'a', 's', ' ', 'i', 's', 't']),
    ])
    def test_tokenizer_wo_args(self, line, expected):
        tokenizer = TokenEncoder().fit(line)

        result = tokenizer.transform([line])[0]
        result = [tokenizer.id2token_[idx] for idx in result]
        assert result == expected

    @pytest.mark.parametrize('special_tokens, expected', [
        ([], ['a', 'b', 'c', 'd', 'e', 'f', 'g']),
        (['a'], ['a', 'b', 'c', 'd', 'e', 'f', 'g']),
        (['ab'], ['ab', 'c', 'd', 'e', 'f', 'g']),
        (['abcdefg'], ['abcdefg']),
        (['abcdefgh'], ['a', 'b', 'c', 'd', 'e', 'f', 'g']),
        (['bc'], ['a', 'bc', 'd', 'e', 'f', 'g']),
        (['efg'], ['a', 'b', 'c', 'd', 'efg']),
        (['ab', 'de'], ['ab', 'c', 'de', 'f', 'g']),
        (['abc', 'cde'], ['abc', 'd', 'e', 'f', 'g']),
        (['abcd', 'efg'], ['abcd', 'efg']),
    ])
    def test_tokenizer_with_special_tokens(self, special_tokens, expected):
        tokenizer = TokenEncoder(special_tokens=special_tokens)
        indices = tokenizer.fit_transform(['abcdefg'])[0]
        result = [tokenizer.id2token_[idx] for idx in indices]
        assert result == expected

        joined = tokenizer.inverse_transform([indices])[0]
        assert joined == 'abcdefg'

    @pytest.mark.parametrize('line', ['$', 'hi', '$hi'])
    def test_tokenizer_with_special_special_token(self, line):
        tokenizer = TokenEncoder(special_tokens=['$']).fit(['hi'])

        encoded = tokenizer.transform([line])
        decoded = tokenizer.inverse_transform(encoded)[0]
        assert decoded == line

    @pytest.mark.parametrize('line, expected', [
        ('abcabcde', ['abc', 'abc', 'd', 'e']),
        ('zabcdabce', ['z', 'abc', 'd', 'abc', 'e']),
        ('defabcgabc', ['d', 'e', 'f', 'abc', 'g', 'abc'])
    ])
    def test_tokenizer_with_special_tokens_more_matches(self, line, expected):
        tokenizer = TokenEncoder(special_tokens=['abc']).fit(ALPHABET)

        encoded = tokenizer.transform([line])[0]
        expected = [tokenizer.token2id_[token] for token in expected]
        assert encoded == expected

    @pytest.mark.parametrize('line, expected', [
        ('das', ['das']),
        ('das haus', ['das', 'haus']),
        ('das haus boot', ['das', 'haus', 'boot']),
        ('das  haus   boot', ['das', 'haus', 'boot']),
        (' das haus   ', ['das', 'haus']),
    ])
    def test_tokenizer_split_on_space(self, line, expected):
        tokenizer = TokenEncoder(separator=" ").fit([line])

        encoded = tokenizer.transform([line])[0]
        expected = [tokenizer.token2id_[token] for token in expected]
        assert encoded == expected

    @pytest.mark.parametrize('line', [
        (''),
        ('hi'),
        ('das boot'),
        ('das boot, ein haus. derHund'),
    ])
    def test_detokenize_with_space(self, line):
        tokenizer = TokenEncoder(separator=" ").fit([line])
        encoded = tokenizer.transform([line])
        decoded = tokenizer.inverse_transform(encoded)[0]

        assert decoded == line

    def test_tokenizer_with_unsupported_separator(self):
        with pytest.raises(NotImplementedError):
            TokenEncoder(separator='and')
