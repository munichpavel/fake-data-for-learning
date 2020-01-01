Welcome to fake-data-for-learning's documentation!
==================================================

Interesting fake multivariate data is harder to generate than it should be. Textbooks typically give definitions, two standard examples (multinomial and multivariate normal) and then proceed to proving theorems and propositions. True, one dimensional distributions can be combined, but here as well the source of examples is also sparse, e.g. products of distributions or copulas (typically Gaussian or t-copulas) applied to these 1-d examples.

For machine learning experimentation, it is useful to have an unlimited supply of interesting fake data, where by interesting I mean that we know certain properties of the data and want to test if the algorithm can pick this up. A great potential source of such data is graphical models.

The goal of this package is to make it easy to generate interesting fake data. We start with discrete Bayesian networks (also known as directed graphical models).

.. toctree::
   :maxdepth: 1
   :caption: Contents

   api
   install

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
