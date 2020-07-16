#!/usr/bin/env python3
"""Setup for primpy: Calculations of quantities of the primordial Universe."""
from setuptools import setup, find_packages

version_dict = {}
exec(open('primpy/__version__.py').read(), version_dict)

setup(
    name='primpy',
    version=version_dict['__version__'],
    description="primpy: Calculations for the primordial Universe.",
    long_description=open('README.md').read(),
    keywords="PPS, cosmic inflation, initial conditions for inflation, kinetic dominance",
    author="Lukas Hergt",
    author_email="lh561@mrao.cam.ac.uk",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pyoscode',
    ],
    tests_require=['pytest'],
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 1",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Cosmology",
        "Operating System :: OS Independent",
    ],
)
