#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit high_dimensional_sampling/__version__.py
version = {}
with open(os.path.join(here, 'high_dimensional_sampling', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='high_dimensional_sampling',
    version=version['__version__'],
    description="Python package for the high dimensional sampling challenge of darkmachines.org.",
    long_description=readme + '\n\n',
    author="Joeri Hermans, Bob Stienen",
    author_email='joeri.hermans@doct.uliege.be, b.stienen@science.ru.nl',
    url='https://github.com/darkmachines/high_dimensional_sampling',
    #packages=[
    #    'high_dimensional_sampling',
    #],
    packages=find_packages(exclude=('tests',)),
    package_dir={'high_dimensional_sampling':
                 'high_dimensional_sampling'},
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='high_dimensional_sampling',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'pyyaml', 'numpy', 'pandas', 'matplotlib', 'seaborn'],
         # FIXME: add your package's dependencies to this list
    setup_requires=[
    #    # dependency for `python setup.py test`
        'pytest-runner',
    #    # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
        'numpy==1.15.4',
        'gpyopt',
        'cma',
        'turbo @ git+https://github.com/uber-research/TuRBO.git',
        'openopt'
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
