#!/usr/bin/env python

"""Package setup file.
"""

from setuptools import setup, find_packages

setup(
    name='planeslam',
    version='1.0',
    description='LiDAR-based SLAM using a Planes',
    author='Adam Dai',
    author_email='adamdai97@gmail.com',
    url='https://github.com/adamdai/planeslam',
    packages=find_packages(),
)