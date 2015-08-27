"""stream, ramdomise, sortify, batching taken from Shawn Tan's
theano_toolkit.

"""
import numpy as np


def valid_batch_size(bs, es):
    """Return the required batch size if hold out validation data is
    required.

    Say the user wants a batch size of 64 and a evaluation size of
    1/3; the streamer has to actually return 64 + 32 samples per
    batch, of which 64 can be used for training and 32 for
    validation. This way 32/(32 + 64) = 1/3 of the data is held out
    for validation, as intended by the user.

    Note: Since there might be rounding, the actual evaluation size
    may diverge to some degree from the required one.

    Parameters
    ----------
    bs : int
      The batch size as indicated by the user.

    es : float
      The evaluation size, i.e. the proportion of data used for
      validation.

    Results
    -------
    bs_valid : int
      Batch size for the validation set.

    """
    bs_valid = es / (1. - es) * bs
    bs_valid = np.round(bs_valid).astype(int)
    return bs_valid


def _clip_tokens(lines, max_len_line):
    """Clip samples in ``lines`` to never have the number of tokens
    exceed ``max_len_line``.

    """
    clipped_lines = [line[:max_len_line] for line in lines]
    return clipped_lines


def _pad_batch(batch, pad_left, pad_right, dtype=np.int32):
    """Pad according to length of longest sample.

    This is so that we have a homogeneous numpy array that can be
    worked as batch.

    Note that we pad with one ``pad_left`` to the left and at least
    one ``pad_right`` to the right.

    Parameters
    ----------
    batch : (list of list) or (2d numpy.array) of tokens/indices
      Input data, already tokenized.

    pad_left : int
      Index used to pad to the left.

    pad_right : int
      Index used to pad to the right

    dtype: data type (default=numpy.int32)
      Since a numpy.array is returned, we may have a change of data
      type; use this to control the data type.

    Returns
    -------
    batch_array : 2d numpy.array
      A homogeneous 2d numpy array that is padded once to the left and
      at least once to the right.

    """
    max_len = max(len(x) for x in batch)
    batch_array = pad_right * np.ones((len(batch), 2 + max_len), dtype=dtype)

    for i, x in enumerate(batch):
        batch_array[i, 0] = pad_left
        batch_array[i, 1:len(x) + 1] = x

    return batch_array


def batch_streamer(
        X,
        batch_size,
        eval_size,
        max_len_line,
        pad_left,
        pad_right,
):
    bs_train = int(batch_size)
    bs_valid = valid_batch_size(bs_train, eval_size)
    bs_total = bs_train + bs_valid

    for i in range(0, len(X), bs_total):
        batch = X[i: i + bs_total]
        batch = _clip_tokens(batch, max_len_line)
        batch = _pad_batch(batch, pad_left, pad_right)
        yield batch
