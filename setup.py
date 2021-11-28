#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Tuesday November 2nd 2021

"""


#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Domenico Di Gangi",
    author_email='digangidomenico@gmail.com',
    python_requires='>=3.6',
 
    description="Fork of a tobit model library for python",
    entry_points={
        'console_scripts': [
            'tobit=tobit.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='tobit',
    name='tobit',
    packages=find_packages(include=['tobit', 'tobit.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/digangidomenico/tobit',
    version='0.1.0',
    zip_safe=False,
)
