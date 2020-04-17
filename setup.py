#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import codecs
import re

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


with open('README.md') as readme_file:
    readme = readme_file.read()


install_requirements = [
    'networkx>=2.4',
    'pandas>=0.25',
    'numpy',
    'scikit-learn>=0.21.3',
    'scipy>=1.3',
    'xarray',
    'pypoman'
]


setup(
    name='fake_data_for_learning',
    version=find_version('fake_data_for_learning', '__init__.py'),
    long_description=readme,
    long_description_content_type='text/markdown',
    project_urls={
        "Homepage": "https://munichpavel.github.io/fake-data-for-learning/",
        "Bug Tracker": "https://github.com/munichpavel/fake-data-for-learning/issues",
        "Documentation": "https://munichpavel.github.io/fake-data-docs/html/index.html",
        "Source Code": "https://github.com/munichpavel/fake-data-for-learning/",
    },
    packages=find_packages(include=['fake_data_for_learning']),
    include_package_data=False,
    python_requires='>3.6',
    install_requires=install_requirements,
    license="MIT license",
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    author="Paul Larsen",
    author_email='munichpavel@gmail.com'
   )
