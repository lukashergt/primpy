#!/usr/bin/env python
"""Test for version consistency."""
from packaging import version


def test_version():
    file_version_dict = {}
    with open("primpy/__version__.py") as versionfile:
        exec(versionfile.read(), file_version_dict)
        file_version = file_version_dict['__version__']

    with open('README.rst') as readmefile:
        for line in readmefile:
            if ':Version:' in line:
                readme_version = line.split(':')[2].strip()

    isinstance(version.parse(file_version), version.Version)
    isinstance(version.parse(readme_version), version.Version)
    assert version.parse(file_version) == version.parse(readme_version)
