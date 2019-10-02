# fake-data-for-learning

Interesting fake multivariate data is not as easy to generate as it should be. Textbooks typically give definitions, two examples (multinomial and multivariate normal) and then proceed to proving stuff. One dimensional distributions can be combined, but here as well the source of examples is also sparse, e.g. products of distributions or copulas.

For machine learning experimentation, it is useful to have an unlimited supply of interesting fake data, where by interesting I mean that we know certain properties of the data and want to test if the algorithm can pick this up. A great potential source of such data is graphical models.

The goal of this package is to make it easy to generate interesting fake data. We start with discrete Bayesian networks (also known as directed graphical models).

## Quickstart

* `git clone` the repository and `cd` into the root directory
* Adapt the `.env.example` file and save as `.env`
* Create a virtual environment using ```conda```, ```virtualenv``` or ```virtualenvwrapper``` and add the relevant command to your `.env` file (e.g. if your environment is named ```myenv```, add ```conda activate myenv```, ```source myenv/bin/activate``` or ```workon myenv```)

## Features

* Sampling from discrete Bayesian networks with integer or string outcome values
* Visualization of network a la ```networkx```


See the demo notebook [bayesian-network.ipynb](notebooks/bayesian-network.ipynb) for feature examples.

## Related packages

This package exists because I became tired of googling for existing implementations of how I wanted to generate fake data. In the development process, however, I found other packages that with the similar core functionality (plus other features), such as

* [Pranab Ghosh](https://github.com/pranab)'s [avenir](https://github.com/pranab/avenir)
* The [pgmpy](http://pgmpy.org/index.html) class [```BayesianModelSampling```](http://pgmpy.org/sampling.html#bayesian-model-samplers)


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [munichpavel fork](https://github.com/munichpavel/cookiecutter-pypackage) of the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
