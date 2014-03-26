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


class BaseFitsData(object):
    """Abstract base class for in-memory representations of FITS data.
    
    This is a base class for variouse Data Containers which are intended to
    hold data contained in fits files (maps, scans, etc.).
    
    """
    
    # This should be overwritten by classes inheriting from this one.
    axes = ()

    def __init__(self, data=None, copy=True):
        """Can either be initialized with a raw data array or with None"""
        
        # Dictionary that holds all data other than .data.  This is safe to 
        # be accessed and updated by the user.
        self.field = {}
        # Dictionary with the same keys as field but holds the axes over which
        # a parameter varies.  For instance, the LST variable varies over the
        # 'time' axis.  axes['LST'] should thus be ('time') and
        # shape(field['LST']) should be (ntimes, ).
        self.field_axes = {}
        # To write data to fits you need a fits format for each field.
        self.field_formats = {}
        # Dictionary that holds the history of this data.  It's keys are
        # history entries for hte data.  They must be strings starting with a
        # three digit integer ennumerating the histories.  The corresponding
        # values give additional details, held in a tuple of strings.  The
        # intension is that when merging data, histories must be identical, but
        # details can be merged.
        self.history = History()

        if data is None :
            self.data = ma.zeros(tuple(np.zeros(len(self.axes))), float)
            self.data_set = False
        else :
            self.set_data(data, copy=copy)

    def set_data(self, data, copy=True):
        """Set the data to passed array."""
        # Feel free to play around with the precision.
        self.data = ma.array(data, dtype=np.float64, copy=copy)
        self.data_set = True
        self.dims = np.shape(data)

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
            a_names = (axis_names,)
        else:
            a_names = axis_names
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
            else:
                msg = ("Could not interpret array dtype as a FITS format."
                       " Please explicitly supply *format* argument.")
                raise ValueError(msg)
        
        self._verify_single_axis_names(a_names)
        self.field[field_name] = field_data
        self.field_axes[field_name] = tuple(a_names)
        self.field_formats[field_name] = str(format)

    def _verify_single_axis_names(self, axis_names) :
        axis_indices = []
        temp_axes = list(self.axes)
        for name in axis_names :
            if not name in temp_axes:
                raise ValueError("Field axes must contain only entries from: ",
                                 str(self.axes))
            temp_axes.remove(name)
            axis_indices.append(list(self.axes).index(name))
        sorted = list(axis_indices)
        sorted.sort()
        if not axis_indices == sorted:
            raise ValueError("Field axes must be well sorted.")

    def verify(self):
        """Verifies that all the data is consistant.

        This method should be run every time you muck around with the data
        and field entries.  It simply checks that all the data is consistant
        (axes, lengths etc.).

        Note that even if you know that your DataBlock will pass the verify,
        you still need to verify as this tells the DataBlock that you are done
        messing with the data.  It then sets some internal variables.
        """
        
        if self.data_set :
            self.dims = np.shape(self.data)
        else :
            raise RunTimeError('Data needs to be set before running verify()')

        # Will delete these keys if they are found in 'field', then see if any
        # are left over.
        axes_keys = self.field_axes.keys()
        format_keys = self.field_formats.keys()
        for field_name in self.field.iterkeys() :
            # Check for keys in fields and not in field_axes, the oposite is
            # done outside this loop.
            if ((not self.field_axes.has_key(field_name)) or 
                (not self.field_formats.has_key(field_name))) :
                raise ce.DataError("Dictionaries 'field', 'field_axes' and "
                                   "field_formats must have the same keys.")
            axes_keys.remove(field_name)
            format_keys.remove(field_name)
            # Check all the axes
            axes = self.field_axes[field_name] # for saving keystrokes only
            self._verify_single_axis_names(axes)
            # Check the shape.
            field_data_shape = np.shape(self.field[field_name])
            for ii in range(len(axes)) :
                axis_ind = list(self.axes).index(axes[ii])
                if field_data_shape[ii] != self.dims[axis_ind] :
                    raise ce.DataError("The shape of the data in one of the "
                                       "fields is incompatible with the shape "
                                       "of the main data. field: "+field_name)
            # Check the format string.
            # TODO: This should do something better than just check that there
            # is a string.
            if not type(self.field_formats[field_name]) is str :
                print type(self.field_formats[field_name])
                print self.field_formats[field_name]
                raise ce.DataError("The field_format must be type str. field: "
                                   + field_name)
        # The opposite of the first check in the loop.
        if len(axes_keys) or len(format_keys) :
            raise ce.DataError("Dictionaries 'field', 'field_axes' and "
                               "field_formats must have the same keys.")

    def add_history(self, history_entry, details=()):
        """Adds a history entry."""
        
        self.history.add(history_entry, details=details)

    def print_history(self):
        """print_history function called on self.history."""

        self.history.display()


class History(dict) :
    """Class that contains the history of a piece of data."""

    def add(self, history_entry, details=()) :

        local_details = details
        # Input checks.
        #if len(history_entry) > 70 :
        #    raise ValueError('History entries limited to 70 characters.')
        if type(details) is str :
        #    if len(details) > 70 :
        #        raise ValueError('History details limited to 70 characters.')
            local_details = (details, )
        for detail in details :
            if not type(detail) is str :
                raise TypeError('History details must be a squence of strings'
                                ' or a single string.')
            #if len(detail) > 70 :
            #    raise ValueError('History details limited to 70 characters.')

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
        """Merge this History object with ones passed to this function."""

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
                        raise ValueError("Histories to be merged must have"
                                         " identical keys.")

    def write(self, fname) :
        """Write this history to disk"""

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

def read_history(fname) :
    """Read a History object from file."""

    f = open(fname, 'r')
    try :
        filestring = f.read()
    finally :
        f.close()
    return History(eval(filestring))

def merge_histories(*args) :
    """Merges DataBlock histories.

    This function accepts an arbitray number of History objects (or classes
    containing a history object in a 'history' attribute), and returns a 
    history dictionary that is a merger of them.  History keys must match; 
    details are added."""
    
    if hasattr(args[0], 'history') :
        history = History(args[0].history)
    else :
        history = History(args[0])
    history.merge(*args[1:])
        
    return history


