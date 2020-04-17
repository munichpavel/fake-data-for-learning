Welcome to fake-data-for-learning's documentation!
==================================================

Interesting fake multivariate data is harder to generate than it should be. Textbooks typically give definitions, two standard examples (multinomial and multivariate normal) and then proceed to proving theorems and propositions. True, one dimensional distributions can be combined, but here as well the source of examples is also sparse, e.g. products of distributions or copulas (typically Gaussian or t-copulas) applied to these 1-d examples.

For machine learning experimentation, it is useful to have an unlimited supply of interesting fake data, where by interesting I mean that we know certain properties of the data and want to test if the algorithm can pick this up. A great potential source of such data is graphical models.

In the current release, we generate fake data with discrete Bayesian networks (also known as directed graphical models).

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api

Installation
------------

Install from `pypi`_: ``pip install fake-data-for-learning``

.. _pypi: https://pypi.org/project/fake-data-for-learning/

Note that the methods of `utils.ProbabilityPolytope` that use polytope calculatations to generate conditional probability tables subject to constraints on expectation value uses the non-pure-python library `pypoman`_. See the installation `instructions`_ for external dependencies.

.. _pypoman: https://github.com/stephane-caron/pypoman
.. _instructions: https://github.com/stephane-caron/pypoman#installation

Links
-----

**Website**: https://munichpavel.github.io/fake-data-for-learning

**GitHub Repository**: https://github.com/munichpavel/fake-data-for-learning

**PyPI**: https://pypi.org/project/fake-data-for-learning/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
