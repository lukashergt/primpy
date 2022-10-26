================================================
primpy: calculations for the primordial Universe
================================================
:primpy: calculations for the primordial Universe
:Author: Lukas Hergt
:Version: 2.3.6
:Homepage: https://github.com/lukashergt/primpy


Installation
------------

.. code:: bash

    pip install primpy

.. code:: bash

    git clone https://github.com/lukashergt/primpy
    cd primpy
    python setup.py install --user

You can check that things are working by running the test suite:

.. code:: bash

    python -m pytest
    flake8 primpy tests
    pydocstyle --convention=numpy primpy


Dependencies
~~~~~~~~~~~~

Basic requirements:

- Python 3.6+
- `numpy <https://pypi.org/project/numpy/>`__
- `scipy <https://pypi.org/project/scipy/>`__
- `pyoscode <https://pypi.org/project/pyoscode/>`__

Documentation:

- `sphinx <https://pypi.org/project/Sphinx/>`__
- `numpydoc <https://pypi.org/project/numpydoc/>`__

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
