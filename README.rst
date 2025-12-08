================================================
primpy: calculations for the primordial Universe
================================================
:primpy: calculations for the primordial Universe
:Author: Lukas Hergt
:Version: 2.18.0
:Homepage: https://github.com/lukashergt/primpy
:Documentation: https://primpy.readthedocs.io

.. image:: https://github.com/lukashergt/primpy/actions/workflows/CI.yaml/badge.svg?branch=master
    :target: https://github.com/lukashergt/primpy/actions/workflows/CI.yaml
    :alt: Build Status
.. image:: https://codecov.io/gh/lukashergt/primpy/branch/master/graph/badge.svg?token=USS4K53PY0
    :target: https://codecov.io/gh/lukashergt/primpy
    :alt: Test Coverage
.. image:: https://readthedocs.org/projects/primpy/badge/?version=latest
    :target: https://primpy.readthedocs.io/en/latest
    :alt: Documentation Status
.. image:: https://img.shields.io/pypi/v/primpy?cacheSeconds=86400
    :target: https://pypi.org/project/primpy
    :alt: PyPI - Version



Installation
------------

``primpy`` can be installed via pip

.. code:: bash

    pip install primpy

or via the setup.py

.. code:: bash

    git clone https://github.com/lukashergt/primpy
    cd primpy
    python setup.py install --user

You can check that things are working by running the test suite:

.. code:: bash

    python -m pytest
    flake8 --max-line-length=99 primpy tests
    pydocstyle --convention=numpy primpy


Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python 3.7+
- `numpy <https://pypi.org/project/numpy/>`__
- `scipy <https://pypi.org/project/scipy/>`__
- `pyoscode <https://pypi.org/project/pyoscode/>`__

Documentation:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `sphinx_rtd_theme <https://pypi.org/project/sphinx-rtd-theme/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__
- `matplotlib <https://pypi.org/project/matplotlib/>`__

Tests:

- `pytest <https://pypi.org/project/pytest/>`__
- `flake8 <https://pypi.org/project/flake8/>`__
- `pydocstyle <https://pypi.org/project/pydocstyle/>`__


Citation
--------

If you use ``primpy`` during your research for a publication, please cite the
following paper:

.. code:: text

    L. T. Hergt, F. J. Agocs, W. J. Handley, M. P. Hobson, and A. N. Lasenby,
    "Finite inflation in curved space", Phys. Rev. D 106, 063529 (2022),
    https://doi.org/10.1103/PhysRevD.106.063529

or using the BibTeX:

.. code:: bibtex

    @article{Hergt2022,
        archivePrefix = {arXiv},
        arxivId = {2205.07374},
        author = {Hergt, L. T. and Agocs, F. J. and Handley, W. J. and Hobson, M. P. and Lasenby, A. N.},
        doi = {10.1103/PhysRevD.106.063529},
        eprint = {2205.07374},
        issn = {2470-0010},
        journal = {Physical Review D},
        month = {sep},
        number = {6},
        pages = {063529},
        publisher = {American Physical Society (APS)},
        title = {{Finite inflation in curved space}},
        url = {https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.063529},
        volume = {106},
        year = {2022}
    }


Contributing
------------
There are many ways you can contribute via the `GitHub repository
<https://github.com/lukashergt/primpy>`__.

- You can `open an issue <https://github.com/lukashergt/primpy/issues>`__ to
  report bugs or to propose new features.
- Pull requests are very welcome. Note that if you are going to propose major
  changes, be sure to open an issue for discussion first, to make sure that
  your PR will be accepted before you spend effort coding it.

