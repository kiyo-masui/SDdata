======
SDdata
======

Formats and conversions for data from single dish radio telescopes.

This package contains various data containers and representations for data 
from single dish radio telescopes, as well as support for reading them to and
from disk in FITS and HDF5 formats.


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
