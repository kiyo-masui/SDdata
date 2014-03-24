"""
Tools and representations of the SDFITS data format.

.. currentmodule:: sddata.sdfits

This module provides tools and representations for the SDFITS data format.
This includes tools for reading and writing SDFITS files, representing them in
memory in a convenient manner, as well manipulating them while enforcing the
SDFITS format.

Unfortunately this module is not currently very general and has been tailored
to 'spectrometer' style data (i.e. spectra vs time) from the Green Bank
Telescope's old spectrometer.

Classes
=======

.. autosummary::
   :toctree: generated/

    BaseFitsBlock
    DataBlock
    Reader
    Writer

"""


import numpy as np
import numpy.ma as ma


