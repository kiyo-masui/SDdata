"""
Tools and representations of the SDFITS data format.

.. currentmodule:: sddata.fits

This module provides tools and representations for the SDFITS data format.
This includes tools for reading and writing SDFITS files, representing them in
memory in a convenient manner, as well manipulating them while enforcing the
SDFITS format.

Unfortunately this module is not currently very general and has been tailored
to 'spectrometer' style data (i.e. spectra vs time) from the Green Bank
Telescope's old spectrometer.

Base Classes and Infrastructure
===============================

.. autosummary::
   :toctree: generated/

    BaseFitsBlock
    Field
    History
    merge_histories
    DataError


Spectrometer Type Data
======================

.. autosummary::
   :toctree: generated/
    
    SpecBlock
    SpecReader
    SpecWriter

"""


import numpy as np
import numpy.ma as ma



# Containers
# ==========

class BaseFitsBlock(object):
    """Abstract base class for in-memory representations of FITS data.
    
    This is a base class for various Data Containers which are intended to
    hold data contained in fits files (maps, scans, etc.).
    
    Attributes
    ----------
    axes
    data
    field
    field_axes
    field_formats
    history

    Methods
    -------
    set_field
    verify
    add_history
    print_history

    """
    
    # This should be overwritten by classes inheriting from this one.
    @property
    def axes(self):
        """Names for the axes of :attr:`~BaseFitsData.data`.

        """
        return ()
    
    @property
    def data(self):
        """The data.
        """
        return self._data

    @property
    def field(self):
        return self._field

    @property
    def history(self):
        return self._history

    def __init__(self, data, copy=True):
        """Can either be initialized with a raw data array or with None"""
        
        # Dictionary that holds all data other than .data.  This is safe to 
        # be accessed and updated by the user.
        self._field = {}

        self._history = History()

        if data.ndim != len(self.axes):
            raise ValueError("Data incompatible with axes: %s."
                             % str(self.axes))
        self._data = ma.array(data, dtype=np.float64, copy=copy)

    def set_field(self, field_name, field_data, axis_names=(), format=None):
        """Set field data to be stored.

        Note that these operation can also be done by accessing the 'field' and
        'field_axes' dictionaries directly, but using this function combines a
        few operations that go together.  It also does some sanity checks.
        Using this function is safer.

        Arguments are the field name (like 'CRVAL2', or 'SCAN'), field data
        (numpy array or appropriate length according to axis_names), axis_names
        (tuple of names like ('time', ) or ('pol',) or simply () for 0D data),
        and finally a fits format string (like '1E' or '10A', see fits
        documentation).
        """

        field_data = np.array(field_data)
        if type(axis_names) is str:
            axis_names = (axis_names,)
        if not format:
            if field_data.dtype == np.float64:
                format = 'D'
            elif field_data.dtype == np.float32:
                format = 'E'
            elif field_data.dtype == np.int16:
                format = 'I'
            elif field_data.dtype == np.int32:
                format = 'J'
            elif field_data.dtype == np.int64:
                format = 'K'
            elif field_data.dtype == np.complex64:
                format = 'C'
            elif field_data.dtype == np.complex128:
                format = 'M'
            elif str(field_data.dtype)[:2] == '|S':
                l = int(str(field_data.dtype)[2:])
                format = '%dA' % l
            else:
                msg = ("Could not interpret array dtype as a FITS format."
                       " Please explicitly supply *format* argument.")
                raise ValueError(msg)
        
        field = Field(field_data, tuple(axis_names), str(format))
        try:
            self._verify_field(field)
        except DataError as e:
            raise ValueError(e.args[0])
        self.field[field_name] = field

    def _verify_field(self, field):
        axis_names = field.axes
        format = field.format

        # Check the axis names.
        axis_indices = []
        temp_axes = list(self.axes)
        for name in axis_names :
            if not name in temp_axes:
                raise DataError("Field axes must contain only entries from: %s."
                                % str(self.axes))
            temp_axes.remove(name)
            axis_indices.append(list(self.axes).index(name))
        sorted = list(axis_indices)
        sorted.sort()
        if not axis_indices == sorted:
            raise DataError("Field axes must be well sorted.")
        
        # Check the shape.
        field_data_shape = field.shape
        for ii in range(len(axis_names)) :
            axis_ind = list(self.axes).index(axis_names[ii])
            if field_data_shape[ii] != self.data.shape[axis_ind] :
                raise DataError("Shape of field incompatible with the shape"
                                " of data.")
        # Check the format string.
        # TODO: This should do something better than just check that there
        # is a string.
        if not type(field.format) is str :
            raise DataError("The field format must be a string.")


    def verify(self):
        """Verifies that all the data is consistent.

        This method should be run every time you muck around with the data
        and field entries.  It simply checks that all the data is consistent
        (axes, lengths etc.).

        Note that even if you know that your DataBlock will pass the verify,
        you still need to verify as this tells the DataBlock that you are done
        messing with the data.  It then sets some internal variables.
        """
        
        for field_name, field in self.field.items():
            try:
                self._verify_field(field)
            except DataError as e:
                e.args[0] = e.args[0] + " Field name: %s." % field_name
                raise e
        
    def add_history(self, history_entry, details=()):
        """Adds a history entry."""
        
        self.history.add(history_entry, details=details)

    def print_history(self):
        """print_history function called on self.history."""

        self.history.display()


class Field(np.ndarray):
    """Represents any 'field' data from a fits file except the "DATA" field.

    Parameters
    ----------
    input_array : numpy array
        Field data.
    axes : tuple of strings
        Names of the axes corresponding to the dimensions of *input_array*.
    format : string
        FITS format string.

    """
    
    # Array creation procedure straight out of numpy documentation:
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, input_array, axes, format):
        obj = np.asarray(input_array).view(cls)
        if len(axes) != obj.ndim:
            raise ValueError("Number of axis names must match dimensions.")
        if not isinstance(format, basestring):
            raise TypeError("*format* must be a FITS data type string.")
        obj._axes = axes
        obj._format = format
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._axes = getattr(obj, '_axes', None)
        self._format = getattr(obj, '_format', None)

    @property
    def axes(self):
        return self._axes

    @property
    def format(self):
        return self._format


# Spectrometer
# ============

class SpecBlock(BaseFitsBlock) :
    """Class that holds an single IF and scan of GBT data.

    Inherits from :class:`BaseFitsBlock`.

    This is the main vessel for storing an transporting GBT data.  This class
    can be used with the fitsGBT.py module to be read and written as a properly
    formatted fits file.  The raw data is accessed and updated through the
    'data' and 'field' attributes of this class and associated hleper
    functions.

    Please remember that when working with the 'data' attribute, that it is a
    numpy MaskedArray class, not a normal numpy array.  Take care to use the
    masked versions of any numpy functions to preserve the mask.  This is
    especially useful for flagging bad data and RFI.

    Attributes
    ----------
    axes

    """
    
    # These are the valid axes that a data field can vary over.  Any other
    # field can vary over only the first three of these.
    @property
    def axes(self):
        "Equals ``('time', 'chan', 'freq')``."
        return ('time', 'chan', 'freq')



# Exceptions
# ==========

class DataError(Exception):
    """Raised for issues with the internal consistency with the data."""


class HistoryError(Exception):
    """Raised for issues with history tracking."""


# History Class
# =============

class History(dict) :
    """Represents the history of a piece of data.
    
    Inherits from :class:`dict`.

    A history entry consists of a key, which is a string, and a list of details
    which are also strings.  The history keys are ordered my prefixing them
    with a zero padded 3 digit integer.

    The intent is to track piece of data through many levels of data
    processing. This tracking is functional even when data with similar
    histories are merged into a derived data product.
    
    Methods
    -------
    add
    merge
    write
    display

    """

    def add(self, history_entry, details=()):
        """Add a history entry.
        
        Parameters
        ----------
        history_entry : str
            Describes the event of the history
        details : sequence
            Strings giving various details about the history entry.

        """

        local_details = details
        if type(details) is str :
            local_details = (details, )
        for detail in details :
            if not type(detail) is str :
                raise TypeError('History details must be a sequence of strings'
                                ' or a single string.')
        n_entries = len(self)
        # '+' operator performs input type check.
        hist_str = ('%03d: ' % n_entries) + history_entry
        self[hist_str] = tuple(local_details)

    def display(self) :
        """Prints the data history in human readable format."""
    
        history_keys = self.keys()
        history_keys.sort()
        for history in history_keys :
            details = self[history]
            print history
            for detail in details :
                print '    ' + detail

    def merge(self, *args) :
        """Merge this multiple :class:`History` objects into this one.
        
        To be mergable, all :class:`History` objects must have identical
        'entries' (keys) but the 'details' (values) or each entry may differ.

        Parameters
        ----------
        *args : sequence
            Any number of :class:`History` objects or objects with an attribute
            :attr:`history`, which is a :class:`History` object.
        
        """

        for obj in args :
            if hasattr(obj, 'history') :
                thishistory = obj.history
            else :
                thishistory = obj
            for entry, details in thishistory.iteritems() :
                for detail in details :
                    try :
                        if not detail in self[entry] :
                            self[entry] = self[entry] + (detail, )
                    except KeyError :
                        raise HistoryError("Histories to be merged must have"
                                         " identical keys.")

    def write(self, fname) :
        """Write this history to disk."""

        f = open(fname, 'w')
        
        try :
            f.write('{\n')
            keys = self.keys()
            keys.sort()
            for history in keys :
                f.write(repr(history))
                f.write(' : (\n    ')
                for detail in self[history] :
                    f.write(repr(detail))
                    f.write(',\n    ')
                f.write('),\n')
            f.write('}\n')
        finally :
            f.close()

def merge_histories(*args) :
    """Merge :class:`History` objects.
    
    Parameters
    ----------
    *args : sequence
        Any number of :class:`History` objects or objects with an attribute
        :attr:`history`, which is a :class:`History` object.
    
    Returns
    -------
    history : :class:`History`
        A new history that is the merger of the input histories.

    See Also
    --------
    :meth:`History.merge`
    
    """
    
    if hasattr(args[0], 'history') :
        history = History(args[0].history)
    else :
        history = History(args[0])
    history.merge(*args[1:])
        
    return history


