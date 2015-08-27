# A natural language RNN model with scikit-learn interface

Use a Recurrent Neural Network (RNN) with Long Short Term Memory
(LSTM) to train on raw text. The RNN learns each line character-wise
and can then generate new sequences based on its experience.


## Introduction

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


## What does Spaghetto do?

```


layers = [
    (InputLayer, {}),
    (EmbeddingLayer, {'output_size': embedding_size}),
    (GRULayer, {'num_units': num_units}),
    (GRULayer, {'num_units': num_units}),
    (RNNDenseLayer, {'nonlinearity': nonlinearities.identity}),
]

rnn = RNN(
    layers,
    encoder=encoder,
    verbose=1,
    max_len_line=300,
)

rnn.fit(X)

```

  epoch  |  train perpl.  |  valid perpl.  |  train/valid | duration  |  sample
------- | -------------- | -------------- | ------------- | ---------- | --------------
      1    |    14.05023    |    13.94870     |   1.00728 | 108.52s   |  Today, I I whemos haok oth h" I wathrey boeithI honl harkg I vhe theol ont an dtor wuttar7 and, FML
      1     |    3.90302      |   4.03950    |    0.96621 | 114.22s   |  Today, my butel, I girled mirur gomelliendyi't gertsery, witt gloset'ss. Oil_ater, writeth, so deower, octior onreat in al, it lookked a liglried, and, intiouttyrarmiends, oneler, sone bablitoldor mirster, ous,b'ssserp, arriend, antalle woilled lastsicarpedsse of my seecid it really pelletuous." So
      1      |   3.78602     |    3.74330    |    1.01140 | 108.59s  |   Today, I had buy. He walked my date canalbicep)ed looked and gatting. FML
      2     |    2.93258     |    3.01190   |     0.97366 | 103.07s  |   Today, and got pationder, I for cregented into negteching ad into analle, my get. She cocchending into me and feeted for "coeter"itedly siffwen. "U picmpled going me." Me her what reached up. FML
      2     |    2.61560     |    2.66600    |    0.98108 | 111.58s    | Today, , was wasted to pooplems uf of going was a keckressort. My hort fordoe to compleral lifzor rivized tomant, so, so had to the sast was pot-to hard down, while was werested to last under to free that getfinwer receptly younger. FML
...

## Getting started

* Have a look at this [annotated notebook](http://nbviewer.ipython.org/github/BenjaminBossan/spaghetto/blob/master/notebooks/train_fmls.ipynb) to see learn how to get started with Spaghetto.
* If you want to try for yourself but don't have data ready, take a look at [this example](http://nbviewer.ipython.org/github/BenjaminBossan/spaghetto/blob/master/notebooks/train_equations.ipynb) that uses synthetic data.

## Install

Download the repository, install the requirements, and run

```
python setup.py install
```
