#!/usr/bin/env python3
"""Setup for primpy: Calculations of quantities of the primordial Universe."""
from setuptools import setup, find_packages

version_dict = {}
with open("primpy/__version__.py") as versionfile:
    exec(versionfile.read(), version_dict)

setup(
    name='primpy',
    version=version_dict['__version__'],
    description="primpy: Calculations for the primordial Universe.",
    long_description=open('README.rst').read(),
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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
    ],
)
