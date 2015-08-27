from setuptools import find_packages
from setuptools import setup

version = '0.0.1'
description = ('A natural RNN language model with '
               'scikit-learn interface')

install_requires = [
    'numpy',
    'scipy',
    'Theano',
    'Lasagne',
    'tabulate',
]

tests_require = [
    'pytest',
    'pytest-cov',
    'mock',
]

setup(
    name='spaghetto',
    version=version,
    description=description,
    author='Benjamin Bossan',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={'testing': tests_require},
)
