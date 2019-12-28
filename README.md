# fake-data-for-learning

Interesting fake multivariate data is harder to generate than it should be. Textbooks typically give definitions, two standard examples (multinomial and multivariate normal) and then proceed to proving theorems and propositions. True, one dimensional distributions can be combined, but here as well the source of examples is also sparse, e.g. products of distributions or copulas (typically Gaussian or t-copulas) applied to these 1-d examples.

For machine learning experimentation, it is useful to have an unlimited supply of interesting fake data, where by interesting I mean that we know certain properties of the data and want to test if the algorithm can pick this up. A great potential source of such data is graphical models.

The goal of this package is to make it easy to generate interesting fake data. We start with discrete Bayesian networks (also known as directed graphical models).

## Quickstart

Install from GitHub

```pip install git+https://github.com/munichpavel/fake-data-for-learning```


For local development

* `git clone` the repository and `cd` into the root directory
* Adapt the `.env.example` file and save as `.env`
* Create a virtual environment using ```conda```, ```virtualenv``` or ```virtualenvwrapper``` and add the relevant command to your `.env` file (e.g. if your environment is named ```myenv```, add ```conda activate myenv```, ```source myenv/bin/activate``` or ```workon myenv```)

### Basic usage

The methods and interfaces for `fake_data_for_learning` largely follow those of [scipy](https://scipy.org), e.g. the method `rvs` to generate random samples, and `pmf` for the probability mass function, with extensions to handle non-integer sample values.

Defining and sampling from (discrete) conditional random variables:

```python
import numpy as np
from fake_data_for_learning import BayesianNodeRV, SampleValue

# Gender -> Y
# Define Gender with probability table, node label and value labels
Gender = BayesianNodeRV('Gender', np.array([0.55, 0.45]), values=['female', 'male'])

# Define Y with conditional probability table, node, value and parent labels
pt_YcGender = np.array([
    [0.9, 0.1],
    [0.4, 0.6],
])
Y = BayesianNodeRV('Y', pt_YcGender, parent_names=['Gender'])

# Evaluate probability mass function for given parent values
Y.pmf(0, parent_values={'Gender': SampleValue('male', label_encoder=Gender.label_encoder)})
# 0.4

# Sample from Y given Gender
Y.rvs({'Gender': SampleValue('male', label_encoder=Gender.label_encoder)}, seed=42)
# array([0])
```

Combine into a Bayesian network. Sample and calculate the probability mass function of each sample.

```python
from fake_data_for_learning import FakeDataBayesianNetwork
samples = bn.rvs(size=5)
# Rounding of pmf is only for display purposes
samples['pmf'] = samples[['Gender', 'Y']].apply(lambda sample: round(bn.pmf(sample), 3), axis=1)
```

![docs/graphics/network_sample.png](docs/graphics/network_sample.png)

Visualize the Bayesian network

```python
bn.draw_graph()
```

![docs/graphics/graph.png](docs/graphics/graph.png)

See the demo notebook [notebooks/bayesian-network.ipynb](notebooks/bayesian-network.ipynb) for feature examples.

## Related packages

This package exists because I became tired of googling for existing implementations of how I wanted to generate fake data. In the development process, however, I found other packages with overlapping functionality (plus other features), notably the [pgmpy](http://pgmpy.org/index.html) class [```BayesianModelSampling```](http://pgmpy.org/sampling.html#bayesian-model-samplers).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [munichpavel fork](https://github.com/munichpavel/cookiecutter-pypackage) of the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
