# A natural language RNN model with scikit-learn interface

Use a Recurrent Neural Network (RNN) with Long Short Term Memory
(LSTM) to train on raw text. The RNN learns each line character-wise
and can then generate new sequences based on its experience.

## Source

The original idea is from [Andrej Karpathy](https://github.com/karpathy/char-rnn).
See this [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

However, the code is in Lua/Torch.  Later, Shawn Tan ported Karpathy's
code to Theano and python
([link](https://github.com/shawntan/theano-nlp)).  More details in
this [blog
post](https://blog.wtf.sg/2015/05/29/generating-singlish-with-lstms/).

This project refactors the code and wraps it with an interface that is
similar to [scikit-learn's](http://scikit-learn.org/) or
[nolearn's](https://github.com/dnouri/nolearn). In addition, some nice
features have been added; for instance, you will see some nice
progress information when fitting the net. Some design choices have
been inspired by and some code been taken from the [Lasagne
project](https://github.com/Lasagne/Lasagne).

## Install

Download the repository, install the requirements, and run

```
python setup.py install
```
