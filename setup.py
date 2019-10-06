#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


requirements = [
    'matplotlib',
    'pandas',
    'networkx',
    'numpy',
]

setup_requirements = [
    'pytest-runner',
    # TODO(munichpavel): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='fake_data_for_learning',
    version='0.1.0',
    author="Paul Larsen",
    author_email='munichpavel@gmail.com',
    url='https://github.com/munichpavel/fake-data-for-learning',
    packages=find_packages(include=['fake_data_for_learning']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='fake_data_for_learning',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
    	'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
