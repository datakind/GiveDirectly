#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='housemapper',
    version='0.1.0',
    description='Map poverty in developer countries by identifying house hold roofing materials',
    long_description=readme + '\n\n' + history,
    author='Lluis Canet',
    author_email='lluis.canet@dadrin.com',
    url='https://github.com/lluiscanet/housemapper',
    packages=[
        'housemapper',
    ],
    package_dir={'housemapper':
                 'housemapper'},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='housemapper',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
