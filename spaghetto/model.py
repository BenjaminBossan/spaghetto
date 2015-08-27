"""Refactorization and wrapping of Shawn Tan's theano-nlp."""
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import re
import sys
import time

from lasagne.layers import get_all_params
from lasagne.layers import get_output
from lasagne.layers import InputLayer
from lasagne.layers import EmbeddingLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import rmsprop
import numpy as np
import theano
import theano.tensor as T

from spaghetto.dataio import batch_streamer
from spaghetto.utils import PrintLog

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class RNN(object):
    def __init__(
        self,
        layers,
        encoder,
        updater=rmsprop,
        batch_size=32,
        streamer=batch_streamer,
        max_epochs=10,
        patience=None,
        improvement_threshold=1.,
        eval_size=0.2,
        verbose=0,
        max_len_line=79,
    ):
        """Class for training and sampling a sequence RNN model

        Parameters
        ----------
        layers : TODO

        encoder : A tokenizer and encoder:
          A spaghetto.utils.TokenEncoder fitted on the data should do
          the job most of the time.
          In general, the encoder has to be an object with two
          methods, ``transform``, which tokenizes a string to a list
          of tokens and then translates these tokens into indices, and
          an ``inverse_transform`` method, which translates a list of
          indices back to a list of tokens and joins that list to form
          a string. Ususally, you want to tokenize on a character
          level.
          Furthermore, the encoder should define an ``start_token``
          and an ``end_token`` attribute, which code for the start of
          a sequence or for the end of a sequence, respectively. And
          the encoder should have a ``num_tokens_`` attribute that is
          the total number of tokens in the data.

        updater : function (default=rmsprop)
          Update function for training the net.

        batch_size : int (default=32):
          Batch size for training data.

        streamer : callable (default=batch_streamer)
          A callable that generates batches from input data.

        max_epochs : int (default=10):
          Number of epochs to train. Can be overridden by fit method.

        eval_size : float (default=0.1)
          Size (in proportion to total) of data held back for validation.

        max_len_line : int (default=79)
          Specifies that maximum length of a sample.

        patience : int or None (default=None):
          Number of epochs of no improvement to wait before stoping
          the training process. If None, don't stop early.

        improvement_threshold : float (default=1.)
          Minimum improvement of loss to still qualify as improvement;
          e.g., for 0.99, a loss that improves from 1.00 to 0.991
          still ticks down patience.

        verbose : int (default=0)
          Controls the verbosity of the model; 0 means off.

        """
        # RNN hyper-parameters
        self.layers = layers
        self.updater = updater
        self.batch_size = batch_size

        # training hyper-parameters
        self.max_epochs = max_epochs
        self.eval_size = eval_size
        self.max_len_line = max_len_line
        self.patience = patience
        self.improvement_threshold = improvement_threshold

        # data and encodings
        self.streamer = streamer
        self.encoder = encoder

        # verbosity
        self.verbose = verbose

        if self.patience and not self.eval_size:
            raise ValueError(
                "Patience depends on valid perplexity but eval_size is "
                "set to {}.".format(eval_size))

    def _initialize_layers(self):
        self.layers_ = OrderedDict()

        layer = None
        for i, (layer_factory, layer_kw) in enumerate(self.layers):
            layer_kw = layer_kw.copy()

            if 'name' not in layer_kw:
                layer_kw['name'] = "{}{}".format(
                    layer_factory.__name__.lower().replace("layer", ""), i)

            if layer_kw['name'] in self.layers_:  # pragma: no cover
                raise ValueError(
                    "Two layers with name {}.".format(layer_kw['name']))

            # Any layers that aren't subclasses of InputLayer are
            # assumed to require an 'incoming' paramter.  By default,
            # we'll use the previous layer as input:
            if not issubclass(layer_factory, InputLayer):
                if 'incoming' in layer_kw:
                    layer_kw['incoming'] = self.layers_[
                        layer_kw['incoming']]
                elif 'incomings' in layer_kw:
                    layer_kw['incomings'] = [
                        self.layers_[name] for name in layer_kw['incomings']]
                else:
                    layer_kw['incoming'] = layer

            # For convenience, automatically set the input layer shape
            # and input_var.
            if issubclass(layer_factory, InputLayer):
                if 'shape' not in layer_kw:
                    layer_kw['shape'] = (None, self.max_len_line)
                if 'input_var' not in layer_kw:
                    layer_kw['input_var'] = T.imatrix('X')

            # For convenience, automatically set the input_size of
            # EmbeddingLayer.
            if issubclass(layer_factory, EmbeddingLayer):
                if 'input_size' not in layer_kw:
                    layer_kw['input_size'] = self.encoder.num_tokens_

            # For convenience, automatically set num_units of output
            # layer.
            if i == len(self.layers) - 1:
                if 'num_units' not in layer_kw:
                    layer_kw['num_units'] = self.encoder.num_tokens_

            layer = layer_factory(**layer_kw)
            self.layers_[layer_kw['name']] = layer

    def _create_iter_funcs(self):
        X = T.imatrix('X')
        Y = T.imatrix('Y')
        sx0, sx1 = X.shape  # input shape
        sy0, sy1 = Y.shape  # output shape
        nt = T.iscalar('num tokens')
        inputs = [X, Y, nt]

        output_layer = self.layers_.values()[-1]

        Y_flat = T.reshape(Y, (sy0 * sy1, 1)).flatten()

        # bs x time x num_tokens
        output_train = get_output(output_layer, X, deterministic=False)
        # bs * time x num_tokens
        output_train_flat = T.reshape(
            output_train[:, :sy1, :], (sx0 * sy1, nt))
        output_train_01 = softmax(output_train_flat)
        probs_train = output_train_01[T.arange(sx0 * sy1), Y_flat]
        loss_train = -T.mean(T.log(probs_train))

        # bs x time x num_tokens
        output_valid = get_output(output_layer, X, deterministic=True)
        # bs * time x num_tokens
        output_valid_flat = T.reshape(
            output_valid[:, :sy1, :], (sx0 * sy1, nt))
        output_valid_01 = softmax(output_valid_flat)
        probs_valid = output_valid_01[T.arange(sx0 * sy1), Y_flat]
        loss_valid = T.mean(-T.log(probs_valid))

        pred_reshape = T.reshape(output_valid, (sx0 * sx1, nt))
        pred_softmax = softmax(pred_reshape)
        pred_valid = T.reshape(pred_softmax, (sx0, sx1, nt))

        all_params = get_all_params(output_layer)
        updates = self.updater(loss_train, all_params)

        train_iter = theano.function(inputs, loss_train, updates=updates)
        valid_iter = theano.function(inputs, loss_valid)
        predict_iter = theano.function([X, nt], pred_valid)

        return train_iter, valid_iter, predict_iter

    def initialize(self):
        """Initialize the model.

        Initialization is called automatically the first time the
        ``fit`` method is called.

        """
        if getattr(self, '_initialized', False):
            return

        self._initialize_layers()

        if self.verbose:
            print(" - Compiling functions ...")
        iter_funcs = self._create_iter_funcs()
        self.train_iter_, self.valid_iter_, self.predict_iter_ = iter_funcs
        if self.verbose:
            print("   ... finished compilation.")

        self.train_history_ = []

        self._initialized = True

    def _get_stream(self, X, **kwargs):
        start_idx = self.encoder.transform([self.encoder.start_token])[0][0]
        end_idx = self.encoder.transform([self.encoder.end_token])[0][0]
        stream_kwargs = {
            'batch_size': self.batch_size,
            'eval_size': self.eval_size,
            'max_len_line': self.max_len_line,
            'pad_left': start_idx,
            'pad_right': end_idx,
        }
        stream_kwargs.update(kwargs)
        stream = self.streamer(X, **stream_kwargs)
        return stream

    def fit(
            self,
            X,
            on_nth_batch=50,
            num_epochs=None,
            max_len_sample=None,
    ):
        """Fit model to data.

        X : str or list of str
          The input data. If a string, they are split into a list of
          lines with length ``max_len_line``. If not, it is assumed
          that X is a list of strings. Each item is treated
          independently. Therefore, use strings if the whole strings
          belongs together (for instance a book), and use a list if
          each item in the list is independent of the other items
          (e.g. a list of independent sentences).

        on_nth_batch : int (default=50):
          Call ``on_batch_finished`` each nth batch.
          Note: ``on_batch_finished`` not yet implemented.

        num_epochs : int (default=None):
          Number of epochs to train. If None, the model's
          ``max_epochs`` is used.

        max_len_sample : int (default=None)
          Determines the length of samples that are shown when
          verbosity is greater than 0. If None, ``max_len_line`` is
          used. This has no influence on the length of samples used
          for training the model.

        Results
        -------
        self : RNN
          Returns self.

        """
        self.initialize()

        X_transformed = self.encoder.transform(X)
        if self.train_history_:
            num_epochs_past = self.train_history_[-1]['epoch'] + 1
        else:
            num_epochs_past = 1
        if num_epochs is None:
            num_epochs = self.max_epochs
        if max_len_sample is None:
            max_len_sample = self.max_len_line

        print_log = PrintLog()
        best_valid_perplexity = np.inf
        no_improvement_count = 0
        batch_count = 0
        skipped = 0

        for epoch in xrange(num_epochs):
            tic = time.time()
            train_loss = []
            train_perplexity = []
            valid_loss = []
            valid_perplexity = []

            X_stream = self._get_stream(X_transformed)

            for Xb in X_stream:
                batch_count += 1

                # Yb is derived from Xb
                Yb = Xb[:, 1:]
                Xb = Xb[:, :-1]

                if self.eval_size > 0:
                    Xt, Xv = Xb[:self.batch_size], Xb[self.batch_size:]
                    Yt, Yv = Yb[:self.batch_size], Yb[self.batch_size:]
                else:
                    Xt, Yt = Xb, Yb

                if (self.eval_size > 0) and (len(Xv) < 1):
                    skipped += len(Xt)
                    continue

                loss = self.train_iter_(Xt, Yt, self.encoder.num_tokens_)
                train_loss.append(loss)
                train_perplexity.append(np.exp(loss))

                if self.eval_size > 0:
                    loss = self.valid_iter_(Xv, Yv, self.encoder.num_tokens_)
                    valid_loss.append(loss)
                    valid_perplexity.append(np.exp(loss))

                if batch_count < on_nth_batch:
                    continue

                batch_count = 0
                # WARNING: This is not yet very clean, since all
                # losses are just averaged, even though the last batch
                # is typically smaller and should thus have a lower
                # weight.
                train_mean = np.mean(train_perplexity[-on_nth_batch:])
                if self.eval_size:
                    valid_mean = np.mean(valid_perplexity[-on_nth_batch:])
                else:
                    valid_mean = 0

                if self.verbose:
                    sample = self.sample(1, max_len_line=max_len_sample)[0]
                    sample = re.sub("\\\r", "", sample)
                    sample = re.sub("\\\n", "", sample)
                else:
                    sample = u""

                info = {
                    'epoch': num_epochs_past + epoch,
                    'train_loss': train_mean, # rename
                    'valid_loss': valid_mean, # rename
                    'duration': time.time() - tic,
                    'sample': sample,
                }
                self.train_history_.append(info)

                if self.verbose:
                    print_log(self.train_history_)
                tic = time.time()

            # save best model and early stopping
            if self.eval_size:
                mean_valid_perplexity = np.mean(valid_perplexity)
            else:
                mean_valid_perplexity = 0

            threshold = self.improvement_threshold * best_valid_perplexity
            if (mean_valid_perplexity < threshold) and self.eval_size:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if self.patience and (no_improvement_count >= self.patience):
                    print("Running out of patience.")
                    break

            # update best valid perplexity
            if best_valid_perplexity > mean_valid_perplexity:
                best_valid_perplexity = mean_valid_perplexity

        if self.verbose and skipped:
            print("Skipped {} sample(s) because the batch did not contain "
                  "enough samples for validation.".format(skipped))

        return self

    def predict_proba(self, X, max_len_line=79):
        """Predict probabilities for tokens in X.

        X : str or list of str
          The input data. If a string, they are split into a list of
          lines with length ``max_len_line``. If not, it is assumed
          that X is a list of strings. Each item is treated
          independently. Therefore, use strings if the whole strings
          belongs together (for instance a book), and use a list if
          each item in the list is independent of the other items
          (e.g. a list of independent sentences).

        max_len_line : int (default=79)
          Maximum number of tokens per line.

        Results
        -------
        probas : numpy.array
          A 3D numpy array with shape (batch size x time steps x
          number of different tokens). Each value corresponds to the
          probability of that specific token for the given time and
          sample.

        """
        Xt = self.encoder.transform(X)
        stream = self._get_stream(Xt, eval_size=0, max_len_line=max_len_line)
        probas = []

        for Xb in stream:
            probas.append(self.predict_iter_(Xb, self.encoder.num_tokens_))

        probas = np.vstack(probas)
        return probas

    def _sample_idx_from_probs(self, probs):
        # use high precision, since sum(probs[:-1]) must never exceed 1.
        probs = probs.astype(np.float64)
        probs /= probs.sum()
        idx = np.inf

        while idx >= self.encoder.num_tokens_:
            idx = np.argmax(np.random.multinomial(1, pvals=probs))

        return idx

    def predict(self, X, temperature=1.0, max_len_line=79):
        """Generate predictions from the RNN.

        X : str or list of str
          The input data. If a string, they are split into a list of
          lines with length ``max_len_line``. If not, it is assumed
          that X is a list of strings.

        temperature : float (default=1.)
          Temperature applied to predictions (controls whether samples
          are in the more likely or the more unlikely domain).
          WARNING: Not sure if it works yet. 

        max_len_line : int (default=79)
          Maximum number of tokens per line.

        Results
        -------
        y_pred : list of str
          Returns for each sample the next predicted tokens.

        """
        end_idx = self.encoder.transform([self.encoder.end_token])[0][0]
        probas = self.predict_proba(X, max_len_line)

        y_pred = []
        for line in probas:
            result = []
            for probs in line:
                probs /= temperature
                idx = self._sample_idx_from_probs(probs)
                result.append(idx)
                if idx == end_idx:
                    break
            y_pred.append(result)
        return self.encoder.inverse_transform(y_pred)

    def sample(
        self,
        num_samples,
        temperature=1.,
        primer=None,
        max_len_line=None,
    ):
        """Generate samples from the RNN.

        num_samples : int
          Number of samples to return.

        temperature : float (default=1.)
          Temperature applied to predictions (controls whether samples
          are in the more likely or the more unlikely domain).
          WARNING: Not sure if it works yet. 

        primer : str or None
          A primer for each sample; if None is given, treat each
          sample as if nothing is known about the preceeding
          sequence. If a string, begin each sample with that string.

        max_len_line : int (default=79)
          Maximum number of tokens per line.

        Results
        -------
        samples : list of strings
          Returns a list of samples.

        """
        end_idx = self.encoder.transform([self.encoder.end_token])[0][0]
        if primer is None:
            primer = self.encoder.start_token
        if max_len_line is None:
            max_len_line = self.max_len_line

        samples = []
        for __ in range(num_samples):
            result = [primer]
            result_ids = self.encoder.transform(result)[0]

            while ((len(result_ids) < max_len_line) and
                   (not result_ids[-1] == end_idx)):
                probs = self.predict_iter_(
                    np.array([result_ids], dtype=np.int32),
                    self.encoder.num_tokens_)
                idx = self._sample_idx_from_probs(probs[0, -1])
                result_ids.append(idx)

            result = self.encoder.inverse_transform([result_ids])[0]
            result = result.replace(self.encoder.start_token, "")
            samples.append(result)

        return samples

    def load_params_from(self, source):
        """Load parameters from ``source``, where source may be
        another RNN or a file.

        """
        self.initialize()

        if isinstance(source, str):
            with open(source, 'rb') as f:
                source = pickle.load(f)

        if isinstance(source, RNN):
            source = get_all_params(list(self.layers_.values())[-1])

        success = "Loaded parameters to layer '{}' (shape {})."
        failure = ("Could not load parameters to layer '{}' because "
                   "shapes did not match: {} vs {}.")

        for key, values in source.items():
            layer = self.layers_.get(key)
            if layer is not None:
                for p1, p2v in zip(layer.get_params(), values):
                    shape1 = p1.get_value().shape
                    shape2 = p2v.shape
                    shape1s = 'x'.join(map(str, shape1))
                    shape2s = 'x'.join(map(str, shape2))
                    if shape1 == shape2:
                        p1.set_value(p2v)
                        if self.verbose:
                            print(success.format(
                                key, shape1s, shape2s))
                    else:
                        if self.verbose:
                            print(failure.format(
                                key, shape1s, shape2s))

    def save_params_to(self, fname):
        """Save model parameters to file specified in ``fname``.
        """
        params = OrderedDict()
        for name, layer in self.layers_.items():
            params[name] = (
                [p.get_value() for p in layer.get_params()] +
                [p.get_value() for p in layer.get_params(regularizable=False)]
            )

        with open(fname, 'wb') as f:
            pickle.dump(params, f, -1)
