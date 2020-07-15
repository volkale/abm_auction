# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='abm_auction',
    version='0.0.1',
    description='',
    author='Alexander Volkmann',
    author_email='alexv@gmx.de',
    packages=find_packages(),
    setup_requires=['pytest-runner>=4.2', 'flake8'],
)
