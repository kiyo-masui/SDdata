======
SDdata
======

Formats and conversions for data from single dish radio telescopes.

Installation
============

This package depends on :mod:`numpy`.  To handle data in FITS format, 
:mod:`pyfits` is required.  For HDF5 format data, :mod:`h5py` is required.

This package is installable by the usual methods, either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

It should also be installable directly with `pip` using the command::

	$ pip install [-e] git+ssh://git@github.com/kiyo-masui/SDdata
