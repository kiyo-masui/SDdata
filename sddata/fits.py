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
    DataError
    History
    merge_histories
    HistoryError


Spectrometer Type Data
======================

.. autosummary::
   :toctree: generated/
    
    SpecBlock
    SpecReader
    SpecWriter

"""

import logging

import numpy as np
import numpy.ma as ma
import pyfits

from sddata.file_utils import abbreviate_file_path


# Set the module logger.
logger = logging.getLogger(__name__)


CARD_HIST = 'HIST'
CARD_HIST = 'HISTDET'


# Containers
# ==========

class BaseFitsBlock(object):
    """Abstract base class for in-memory representations of FITS data.
    
    This is a base class for various Data Containers which are intended to
    hold data contained in fits files (maps, scans, etc.). This is not intended
    to represent a whole FITS file or even header-data-unit, rather a logical 
    'block' of data.  Many such blocks may be stored in the same table with
    some natural division, such as by scan number, receiver antenna beams, or
    spectral bands.

    The main issue that this class attempts to resolve is replication of
    meta-data in tables representing multidimensional arrays. In SDFITS, data
    is stored in a one-dimensional table of records, with a new record created
    for each time-sample, polarization channel, noise-cal gate, beam, etc.  As
    such an identical time stamp must be recorded for every permutation of
    polarization, beam, etc. If left in this format when manipulating the data,
    the meta-data is difficult to deal with and is likely to become
    inconsistent.
    
    To address this we decimate the data down to it's minimal representation
    for the multi-dimensional dataset when reading from disk and re-replicate it
    when writing it back to a table in the FITS file.

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
    """Represents any 'field' data from a FITS table except the "DATA" field.

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
    """Represents data from spectrometer type instruments.

    Inherits from :class:`BaseFitsBlock`.
    
    This type of data has axes representing time, channel, and spectral
    frequency (where channel is a catch all for things like polarization and
    noise-cal gate).

    GBT Old Spectrometer FITS
    -------------------------
    Each scan and intermediate frequency (IF) is put into its own separate
    block.  The 'chan' axis is generally length 8 for the 4 polarizations and 2
    noise-cal states.

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


# Unfortunately we have to hard code which fields to read in from the fits
# files and what axes each one should vary over.  This list isn't complete and
# may not be the same for each telescope.

# For a description of the fields: 
# https://safe.nrao.edu/wiki/bin/view/Main/SdfitsDetails
SPEC_FIELDS = { 
   'SCAN' : (),
   'BANDWID' : (),
   'CRPIX1' : (),    # Centre frequency channel
   'CDELT1' : (),    # Frequency channel width
   'OBJECT' : (),
   'TIMESTAMP' : (),
   'OBSERVER' : (),
   'RESTFREQ' : (),
   'DURATION' : (),
   'CTYPE1' : (),
   'CTYPE2' : (),
   'CTYPE3' : (),
   'CRVAL1' : ('time',),    # Band centre frequency
   'DATE-OBS' : ('time', ),
   'LST' : ('time', ),
   'ELEVATIO' : ('time', ),
   'AZIMUTH' : ('time', ),
   'CRVAL2' : ('time', ),
   'CRVAL3' : ('time', ),
   'RA' : ('time', ),
   'DEC' : ('time', ),
   'CRVAL4' : ('chan',),
   'CAL' : ('chan', ),
   'BEAM' : ('cahn',),
   'EXPOSURE' : ('time', 'chan'),
   }


class SpecReader(object):
    """Class that opens a Spectrometer Fits file and reads data.

    This class opens an SDFITS file upon initialization and closes it upon
    deletion.  It contains routines for reading individual scans and IFs from
    the file.  This class reads data but does not store data.  Data is stored
    in :class:`SpecBlock` objects, which are returned by :meth:`read()`.

    The data in the file are subdivided by the 'SCAN' an 'CRVAL1' fields,
    corresponding to separate scans and bands respectively.
    The fields 'BEAM', 'CRVAL4' and 'CAL' (corresponding to receiver pixel,
    polarization and noise-cal gate) are packed into the 'chan' axis of
    individual :class:`SpecBlock` objects.

    Currently, records are assumed to be sorted by time with 'BEAM',
    'CRVAL4', and 'CAL' cycling in a fixed manner withing a single frame.

    Parameters
    ----------
    fname : string
        FITS file name to be read.  The file is assumed to have a certain
        entries and be arranged a certain way corresponding to the GBT
        spectrometer data.
    memmap : bool
        Whether to memory map the file while reading.  This is generally better
        on memory.

    Attributes
    ----------
    scans
    bands

    Methods
    -------
    read

    """

    @property
    def scans(self):
        """Set of scans in the FITS file.
        
        Entries are integers corresponding to the unique values of the 'SCAN'
        field of the data table.

        """
        return self._scans.copy()
    
    @property
    def bands(self):
        """Set of sub-bands in the FITS file.

        Entries are floats corresponding to the unique band centres in Hz of
        each sub-band. In the FITS file these are the unique values of the 'CRVAL1'
        field of the data table, rounded to the nearest kHz.
        
        """

        return self._bands.copy()

    def __init__(self, fname, memmap=False):

        self.verify_ordering = True

        self.fname = fname

        # The passed file name is assumed to be a GBT spectrometer fits file.
        self.hdulist = pyfits.open(self.fname, 'readonly', memmap=memmap)
        if len(self.hdulist) < 2:
            raise DataError("Fits file missing data extension")
        logger.info("Opened GBT fits file: %s" % abbreviate_file_path(fname))

        fits_data = self.hdulist[1].data
        n_records = len(fits_data)
        self._fits_data = fits_data

        try:
            names = fits_data.names
        except AttributeError:
            names = fits_data._names
        self._field_names = names
        
        # Figure out the scan and sub-band content of the file.
        if 'SCAN' in names:
            self._scans_all = fits_data.field('SCAN')
            scans = np.unique(self._scans_all)
            np.sort(scans)
            self._scans = scans
        else:
            self._scans_all = np.zeros(len(n_records))
            self._scans = np.array([0])

        # Round the frequencies to nearest 10 kHz as we only need to tell the
        # difference between one sub-band and the other.
        self._bands_all = np.round(fits_data.field('CRVAL1'), -4)
        bands = np.unique(self._bands_all)
        np.sort(bands)
        self._bands = bands
    
    def get_scan_band_records(self, scan_ind, band_ind) :
        """Gets the records of the fits file that correspond to the
        given scan and sub-band. Reshapes the records to (ntime, nchan).

        """

        thescan = self.scans[scan_ind]
        theband = self.bands[band_ind]
        
        # Find all the records that correspond to this IF and this scan.
        # These indices *should* now be ordered in time, cal (on off)
        # and in polarization, once the IF is isolated.
        inds = np.logical_and(self._bands_all==theband, self._scans_all==thescan)
        records = np.array(self._fits_data[inds])  # Performs copy.
        if 'CAL' in self._field_names:
            ncal = len(np.unique(records['CAL']))
        else:
            ncal = 1
        if 'CRVAL4' in self._field_names:
            npol = len(np.unique(records['CRVAL4']))
        else:
            npol = 1
        if 'BEAM' in self._field_names:
            nbeam = len(np.unique(records['BEAM']))
        else:
            nbeam = 1
        
        # Reform to organize by pol, cal, etc.
        nchan = npol * ncal * nbeam
        ntime = len(records) // nchan
        records.shape = (ntime, nchan)
        
        # Check that cal, pol and beam are properly sorted.
        for the_chan in range(nchan):
            chan_rec = records[:,the_chan]
            if 'CAL' in self._field_names:
                if len(np.unique(chan_rec['CAL'])) != 1:
                    msg = "Cal-states not regularly ordered in file: %s."
                    msg = msg % self.fname
                    raise DataError(msg)
            if 'CRVAL4' in self._field_names:
                if len(np.unique(chan_rec['CRVAL4'])) != 1:
                    msg = "Polarizations not regularly ordered in file: %s."
                    msg = msg % self.fname
                    raise DataError(msg)
            if 'BEAM' in self._field_names:
                if len(np.unique(chan_rec['BEAM'])) != 1:
                    msg = "Beams not regularly ordered in file: %s."
                    msg = msg % self.fname
                    raise DataError(msg)
            if 'DATE-OBS' in self._field_names:
                if len(np.unique(chan_rec['DATE-OBS'])) != ntime:
                    msg = ("Time stamps repeated. Perhaps there are extra"
                           " channels. In file: %s.")
                    msg = msg % self.fname
                    raise DataError(msg)
        for the_time in range(ntime):
            if 'DATE-OBS' in self._field_names:
                if len(np.unique(records[the_time,:]['DATE-OBS'])) != 1:
                    msg = "Time stamps not regularly ordered in file: %s."
                    msg = msg % self.fname
                    raise DataError(msg)
        return records


    def set_history(self, Block):
        """Reads the file history and sets the corresponding Block history."""

        prihdr = self.hdulist[0].header
        # If there is no history, return.
        try:
            ii = prihdr.ascardlist().index_of(card_hist)
        except KeyError:
            return
        n_cards = len(prihdr.ascardlist().keys())
        while ii < n_cards :
            if prihdr.ascardlist().keys()[ii] == card_hist:
                hist_entry = prihdr[ii]
                details = []
            elif prihdr.ascardlist().keys()[ii] == card_detail:
                details.append(prihdr[ii])
            ii = ii + 1
            if ii == n_cards or prihdr.ascardlist().keys()[ii] == card_hist:
                Block.add_history(hist_entry, details)

    def read(self, scans=None, bands=None, beams=None, force_tuple=False,
             IFs=None) :
        """Read in data from the fits file.

        This method reads data from the fits file including the files history
        and basically every peice of data that could be needed.  It is done,
        one scan and one IF at a time.  Each scan and IF is returned in an
        instance of the DataBlock class (defined in another module of this
        package).

        Parameters
        ----------
        scans : tuple of integers
            Which scans in the file to be processed.  A list of 
            integers, with 0 corresponding to the lowest numbered scan.
            Default is all of them.
        bands : tuple of integers
            Which intermediate frequencies (also called frequency windows)
            to process.  A list of integers with 0 coorsponding to the 
            lowest frequency present. Default is all of them.
            TODO : Overlapping frequency windows stiched together somehow.
        force_tuple: By default, if there is only a single output Data
            Block, it is returned not wraped in a tuple, but if we want to
            loop over the output we can force the output to be a tuple,
            even if it only has one element.
        IFs : tuple of integers
            Depricated, use `bands`.

        Returns
        -------
        Instance of the DataBlock class, or a tuple of these instances
        (if asked for multiple scans and IFs).
        """
        
        # `bands` and `IFs` is are two names for the same parameter.
        if not bands is None and IFs is None:
            IFs = bands
        # We want scans and IFs to be a sequence of indicies.
        if scans is None :
            scans = range(len(self.scan_set))
        elif not hasattr(scans, '__iter__') :
            scans = (scans, )
        elif len(scans) == 0 :
            scans = range(len(self.scan_set))
        if IFs is None :
            IFs = range(len(self.IF_set))
        elif not hasattr(IFs, '__iter__') :
            IFs = (IFs, )
        elif len(IFs) == 0 :
            IFs = range(len(self.IF_set))
        if beams is None :
            beams = range(len(self.beam_set))
        elif not hasattr(beams, '__iter__') :
            beams = (beams, )
        elif len(beams) == 0 :
            beams = range(len(self.beam_set))

        
        # Sequence of output DataBlock objects.
        output = ()
        for scan_ind in scans :
            for IF_ind in IFs :
                for beam_ind in beams:
                    # Choose the appropriate records from the file, get that data.
                    inds_sif = self.get_scan_band_beam_inds(scan_ind, IF_ind,
                                                            beam_ind)
                    Data_sif = db.DataBlock(self.fitsdata.field('DATA')[inds_sif])
                    # Masked data is stored in FITS files as float('nan')
                    Data_sif.data[np.logical_not(np.isfinite(
                                       Data_sif.data))] = ma.masked
                    # Now iterate over the fields and add them
                    # to the data block.
                    for field, axis in fields_and_axes.iteritems() :
                        # See if this fits file has the key we are looking for.
                        try:
                            names = self.fitsdata.names
                        except AttributeError:
                            names = self.fitsdata._names
                        if not field in names :
                            continue
                        # First get the 'FITS' format string.
                        field_format = self.hdulist[1].columns.formats[
                                    self.hdulist[1].columns.names.index(field)]
                        if axis :
                            # From the indices in inds_sif, we only need a
                            # subset: which_data will subscript inds_sif.
                            temp_data = self.fitsdata.field(field)[inds_sif]
                            # For reshaping at the end.
                            field_shape = []
                            for ii, single_axis in enumerate(Data_sif.axes[0:-1]) :
                                # For each axis, slice out all the data except the
                                # stuff we need.
                                which_data = [slice(None)] * 3
                                if single_axis in axis :
                                    field_shape.append(Data_sif.dims[ii])
                                else :
                                    which_data[ii] = [0]
                                temp_data = temp_data[tuple(which_data)]
                            temp_data.shape = tuple(field_shape)
                            Data_sif.set_field(field, temp_data, axis, field_format)
                        else :
                            Data_sif.set_field(field, self.fitsdata.field(field)
                                [inds_sif[0,0,0]], axis, field_format)
                    if hasattr(self, 'history') :
                        Data_sif.history = db.History(self.history)
                    else :
                        self.history =bf.get_history_header(self.hdulist[0].header)
                        #self.set_history(Data_sif)
                        fname_abbr = ku.abbreviate_file_path(self.fname)
                        self.history.add('Read from file.', ('File name: ' + 
                                             fname_abbr, ))
                        Data_sif.history = db.History(self.history)
                    Data_sif.verify()
                    output = output + (Data_sif, )
        print 'Read finished.'
        if len(output) == 1 and not force_tuple :
            return output[0]
        else :
            return output

    def __del__(self):
        self.hdulist.close()


class Writer():
    """Class that writes data back to fits files.

    This class acculumates data stored in DataBlock objects using the
    'add_data(DataBlock)' method.  Once the user has added all the data, 
    she can then call the 'write(file_name)' method to write it to file.
    """
    
    def __init__(self, Blocks=None):
        
        self.first_block_added = True
        self.field = {}
        self.formats = {}
        if not Blocks is None:
            self.add_data(Blocks)

    def add_data(self, Blocks) :
        """Interface for adding DataBlock objects to the Writter.
        
        This method can be passed either a single DataBlock object or any
        sequence of DataBlock objects.  They will all be added to the Writer's
        data which can eventually be written as a fits file.
        """

        if not hasattr(Blocks, '__iter__'):
            self._add_single_block(Blocks)
        else :
            for Block in Blocks:
                self._add_single_block(Block)

    def _add_single_block(self, Block):
        """Adds all the data in a DataBlock Object to the Writer such that it
        can be written to a fits file eventually."""
        
        Block.verify()
        # Merge the histories
        if self.first_block_added :
            self.history = db.History(Block.history)
        else :
            self.history = db.merge_histories(self.history, Block)
        # Some dimensioning and such
        dims = tuple(Block.dims)
        n_records = dims[0]*dims[1]*dims[2]
        block_shape = dims[0:-1]
        # For now automatically determine the format for the data field.
        data_format = str(dims[-1]) + 'E'
        if self.first_block_added :
            self.data_format = data_format
        elif self.data_format != data_format :
            raise ce.DataError('Data shape miss match: freq axis must be same'
                               ' length for all DataBlocks added to Wirter.')

        # Copy the reshaped data from the DataBlock
        data = Block.data.filled(np.nan)
        if self.first_block_added :
            self.data = data.reshape((n_records, dims[3]))
        else :
            self.data = np.concatenate((self.data, data.reshape((
                                        n_records, dims[3]))), axis=0)

        # Now get all stored fields for writing out.
        for field, axes in Block.field_axes.iteritems() :
            # Need to expand the field data to the full ntime x npol x ncal
            # length (with lots of repitition).  We will use np broadcasting.
            broadcast_shape = [1,1,1]
            for axis in axes :
                axis_ind = list(Block.axes).index(axis)
                broadcast_shape[axis_ind] = dims[axis_ind]
            # Allowcate memory for the new full field.
            data_type = Block.field[field].dtype
            field_data = np.empty(block_shape, dtype=data_type)
            # Copy data with the entries, expanding dummy axes.
            field_data[:,:,:] = np.reshape(Block.field[field],
                                                 broadcast_shape)
            if self.first_block_added :
                self.field[field] = field_data.reshape(n_records)
                self.formats[field] = Block.field_formats[field]
            else :
                self.field[field] = np.concatenate((self.field[field],
                                        field_data.reshape(n_records)), axis=0)
                if self.formats[field] != Block.field_formats[field] :
                    raise ce.DataError('Format miss match in added data blocks'
                                       ' and field: ' + field)
        self.first_block_added = False

    def write(self, file_name):
        """Write stored data to file.
        
        Take all the data stored in the Writer (from added DataBlocks) and
        write it to a fits file with the passed file name.
        """

        # Add the data
        Col = pyfits.Column(name='DATA', format=self.data_format, 
                            array=self.data)
        columns = [Col,]
        
        # Add all the other stored fields.
        for field_name in self.field.iterkeys() :
            Col = pyfits.Column(name=field_name,
                                format=self.formats[field_name],
                                array=self.field[field_name])
            columns.append(Col)
        coldefs = pyfits.ColDefs(columns)
        # Creat fits header data units, one for the table and the mandatory
        # primary.
        tbhdu = pyfits.new_table(coldefs)
        prihdu = pyfits.PrimaryHDU()
        # Add the write history.
        fname_abbr = ku.abbreviate_file_path(file_name)
        self.history.add('Written to file.', ('File name: ' + fname_abbr,))
        # Add the history to the header.
        bf.write_history_header(prihdu.header, self.history)

        # Combine the HDUs and write to file.
        hdulist = pyfits.HDUList([prihdu, tbhdu])
        hdulist.writeto(file_name, clobber=True)
        print 'Wrote data to file: ' + fname_abbr


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


