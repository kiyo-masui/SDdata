"""Unit tests for :mod:`fits` module."""

import unittest
import os
from os import path

import numpy as np
import numpy.ma as ma

import sddata.fits as fits


data_path = path.dirname(path.realpath( __file__ )) + '/data/'


# Tests for data container
# ------------------------

ntime = 7
npol = 4
ncal = 2
nfreq = 10
shape = (ntime, npol*ncal, nfreq)

class TestSpecBlockSetup(unittest.TestCase):

    def test_raises_dims(self):
        data_arr = np.zeros((ntime, npol, ncal, nfreq))
        self.assertRaises(ValueError, fits.SpecBlock, data_arr)

    def test_copies(self):
        data_arr = np.zeros((ntime, npol*ncal, nfreq))
        data = fits.SpecBlock(data_arr)
        data_arr[1,2,3] = 5.
        self.assertEqual(data.data[1,2,3], 0)


class TestSpecBlockFields(unittest.TestCase):

    def setUp(self):
        data_arr = np.arange(ntime*npol*ncal*nfreq)
        data_arr.shape = (ntime, npol*ncal, nfreq)
        self.data = fits.SpecBlock(data_arr)

    def test_set_get_field(self):
        crval4 = [-5, -5, -6, -6, -7, -7, -8, -8]
        cal = ['T', 'F'] * 4
        time = np.arange(ntime) * 1.234
        self.data.set_field('CRVAL4', crval4, ('chan',), 'I')
        self.data.set_field('CAL', cal, ('chan',), '1A')
        self.data.set_field('TIME', time, ('time',), 'E')
        self.assertTrue(np.all(self.data.field['CRVAL4'] == crval4))
        self.assertTrue(np.all(self.data.field['CAL'] == cal))
        self.assertTrue(np.allclose(self.data.field['TIME'], time))

    def test_set_invalid_shape(self):
        crval4 = [-5, -5, -6, -6, -7, -7, -8, -8, -9]
        self.assertRaises(ValueError, self.data.set_field, 'CRVAL4', crval4, 
                          ('chan',), 'I')
        time = np.arange(ntime) * 1.234
        time.shape = (1, ntime)
        self.assertRaises(ValueError, self.data.set_field, 'TIME', time, 
                          ('time',), 'E')

    def test_set_invalid_axes(self):
        cal = ['T', 'F'] * 4
        time = np.arange(ntime) * 1.234
        self.assertRaises(ValueError, self.data.set_field, 
                'CAL', cal, ('chn',), '1A')

    def test_detect_format(self):
        crval4 = [-5, -5, -6, -6, -7, -7, -8, -8]
        cal = ['T', 'F'] * 4
        time = np.arange(ntime, dtype=float) * 1.234
        self.data.set_field('CRVAL4', crval4, ('chan',))
        self.data.set_field('CAL', cal, ('chan',))
        self.data.set_field('TIME', time, ('time',))
        self.assertEqual(self.data.field['CRVAL4'].format, 'K')
        self.assertEqual(self.data.field['CAL'].format, '1A')
        self.assertEqual(self.data.field['TIME'].format, 'D')
    
    def test_multiD_field(self):
        stuff = np.arange(ntime * npol * ncal)
        stuff.shape = (ntime, npol * ncal)
        self.data.set_field('STUFF', stuff, ('time', 'chan'))
        stuff.shape = (ntime * npol, ncal)
        self.assertRaises(ValueError, self.data.set_field, 'STUFF1', stuff,
                          ('time', 'chan'))


class TestHistory(unittest.TestCase) :
    
    def setUp(self) :
        self.data = fits.SpecBlock(np.zeros((2,2,2)))
        self.hist_str = 'For example, saved to file.'
        self.hist_detail = ('the file name', 'the date perhaps')

    def test_add_history(self) :
        # The basics:
        self.data.add_history(self.hist_str, self.hist_detail)
        self.assertTrue(self.data.history.has_key('000: '+self.hist_str))
        self.assertTrue(self.data.history['000: '+self.hist_str][0] == 
                        self.hist_detail[0])

    def test_add_hist_no_detail(self) :
        self.data.add_history(self.hist_str)
        self.assertTrue(self.data.history.has_key('000: '+self.hist_str))
        self.assertEqual(len(self.data.history['000: '+self.hist_str]), 0)

    def test_add_string_detail(self) :
        self.data.add_history(self.hist_str, self.hist_detail[0])
        self.assertTrue(self.data.history['000: '+self.hist_str][0] == 
                        self.hist_detail[0])

    def test_merge_histories(self) :
        # Basic tests
        self.data.add_history(self.hist_str, self.hist_detail)
        SecondDB = fits.SpecBlock(np.zeros((2,2,2)))
        second_details = ('second file name', )
        SecondDB.add_history(self.hist_str, second_details)
        merged = fits.merge_histories(self.data, SecondDB)
        self.assertEqual(len(merged.keys()), 1)
        self.assertTrue(second_details[0] in merged['000: '+self.hist_str])
        self.assertTrue(self.hist_detail[0] in merged['000: '+self.hist_str])
        self.assertEqual(len(merged['000: '+self.hist_str]), 3)
        # Enforce matching.
        ThirdDB = fits.SpecBlock(np.zeros((2,2,2)))
        ThirdDB.add_history(self.hist_str, self.hist_detail)
        ThirdDB.add_history('Read from file.', self.hist_detail)
        self.assertRaises(fits.HistoryError, fits.merge_histories, SecondDB, ThirdDB)

    def test_merge_multiple_histories(self) :
        entry1 = 'Read from file.'
        entry2 = 'Saved to file.'
        DBs = ()
        n = 10
        for ii in range(n) :
            tempDB = fits.SpecBlock(np.zeros((2,2,2)))
            tempDB.add_history(entry1, 'filename: ' + str(ii))
            tempDB.add_history(entry2, 'saved filename not iterated')
            DBs = DBs + (tempDB, )
        merged = fits.merge_histories(*DBs)
        self.assertEqual(len(merged['000: '+entry1]), n)
        self.assertEqual(len(merged['001: '+entry2]), 1)


# Tests for SDFITS IO
# -------------------

# This fits file generated by the script make_test_GBT_fits_file.py
testfile_gos = path.join(data_path, 'GBT_old_spec.sdfits')
# This file has known properties:
bands_gos = (695010000, 725010000)
scans_gos = (17, 18)
pol_set = (-5, -7, -8, -6)
cal_set = ('T', 'F')
nband_gos = len(bands_gos)
nscan_gos = len(scans_gos)
ntime_gos = 10
npol_gos = len(pol_set)
ncal_gos = len(cal_set)
nchan_gos = npol_gos * ncal_gos
nfreq_gos = 2048
# Subset of fields known to be present in test data.
fields_gos = ['SCAN', 'OBSERVER', 'RESTFREQ', 'CTYPE1', 'CRVAL1', 'DATE-OBS',
              'CRVAL2', 'CRVAL3', 'CRVAL4', 'CAL', 'EXPOSURE']


class TestReaderInit(unittest.TestCase) :
    
    def setUp(self):
        self.reader = fits.SpecReader(testfile_gos)

    def test_gets_bands(self):
        for ii in range(len(bands_gos)):
            self.assertAlmostEqual(self.reader.bands[ii], bands_gos[ii])

    def test_gets_scans(self):
        for ii in range(len(scans_gos)):
            self.assertEqual(self.reader.scans[ii], scans_gos[ii])

    def tearDown(self):
        del self.reader


class TestReaderGetIFScanInds(unittest.TestCase) :
    
    def setUp(self) : 
        self.reader = fits.SpecReader(testfile_gos)
        bands_all = np.array(self.reader._fits_data.field('CRVAL1'))
        bands_all = np.round(bands_all, 0)
        self.bands_all  = np.array(bands_all, int)
        self.scans_all = np.array(self.reader._fits_data.field('SCAN')).copy()

    def test_gets_records(self) :
        for scan_ind in range(2) :
            for band_ind in range(2) :
                records = self.reader.get_scan_band_records(scan_ind, band_ind)
                # Verify we got all of them.
                self.assertEqual(records.size, npol_gos * ncal_gos * ntime_gos)
                # Verify they are all unique.
                self.assertEqual(len(np.unique(records)),
                                 npol_gos * ncal_gos * ntime_gos)

    def test_reforms_records(self) :
        """Test reshaping of indicies to time x pol x cal."""
        
        # Get the inds of a scan and IF and use them to get some data.
        records = self.reader.get_scan_band_records(1, 1)
        LST = records['LST']
        pol = records['CRVAL4']
        cal = records['CAL']
        # Test that the indicies have the proper shape
        shape_expected = (ntime_gos, npol_gos * ncal_gos)
        self.assertEqual(shape_expected, records.shape)
        # Make sure that LST is constant of indicies 1,2.  Etc. for pol, cal.
        aLST = np.unique(LST[0,:])
        self.assertEqual(len(aLST), 1)
        apol = np.unique(pol[:,0])
        self.assertEqual(len(apol), 1)
        acal = np.unique(cal[:,0])
        self.assertEqual(len(acal), 1)

    def test_checks_cal_order(self) :
        """Puts cals out of order and check if exception is raised."""

        # Mess up the cals in one of the scans, IFs
        fits_data = np.array(self.reader._fits_data)
        self.reader._fits_data = fits_data
        fits_data['CAL'][10] = 'T'
        fits_data['CAL'][11] = 'T'
        self.assertRaises(fits.DataError, self.reader.get_scan_band_records,
                          0, 1)

    def test_checks_pol_order(self) :
        """Puts pols out of order and check if exception is raised."""

        # Mess up the cals in one of the scans, IFs
        fits_data = np.array(self.reader._fits_data)
        self.reader._fits_data = fits_data
        fits_data['CRVAL4'][3] = '-8'
        fits_data['CRVAL4'][4] = '-8'
        self.assertRaises(fits.DataError, self.reader.get_scan_band_records,
                          0, 0)

    def test_checks_pol_order(self) :
        """Puts pols out of order and check if exception is raised."""

        # Mess up the cals in one of the scans, IFs
        fits_data = np.array(self.reader._fits_data)
        self.reader._fits_data = fits_data
        fits_data['DATE-OBS'][3] = fits_data['DATE-OBS'][101]
        self.assertRaises(fits.DataError, self.reader.get_scan_band_records,
                          0, 0)

    def tearDown(self) :
        del self.reader


class TestReads(unittest.TestCase) :
    """Some basic test for some know properties of the data in the test fits
    file."""
    
    def setUp(self) :
        self.reader = fits.SpecReader(testfile_gos)
        self.block = self.reader.read(0, 0)[0]
        self.block.verify()

    def test_reads_valid_data(self):
        s = self.block.data.shape
        self.assertEqual(s, (ntime_gos, nchan_gos, nfreq_gos))
        
    def test_feilds(self) :
        self.assertEqual(self.block.field['SCAN'], scans_gos[0])
        self.assertEqual(round(self.block.field['CRVAL1'][0]/1e6), 695)
        for ii in range(npol_gos) :
            self.assertEqual(pol_set[ii], self.block.field['CRVAL4'][2*ii])
        for ii in range(ncal_gos) :
            self.assertEqual(cal_set[ii], self.block.field['CAL'][ii])
        self.assertEqual(self.block.field['CRVAL1'].format, '1D')
        self.assertEqual(self.block.field['CRVAL4'].format, '1I')
        # Multi Dimensional field read.
        self.assertEqual(self.block.field['EXPOSURE'].shape, 
                         (ntime_gos, nchan_gos))
        self.assertEqual(self.block.field['EXPOSURE'].axes, 
                         ('time', 'chan'))

    def tearDown(self) :
        del self.reader
        del self.block


class TestMultiRead(unittest.TestCase) :
    """Test that each scan and IF is read exactly 1 time by default."""
    
    def test_multiple_reads(self) :
        nscan = len(scans_gos)
        nband = len(bands_gos)
        reader = fits.SpecReader(testfile_gos)
        blocks = reader.read()
        
        self.assertEqual(len(blocks), nscan_gos*nband_gos)
        # Lists multiplied by two because each scan shows up in 2 IFs.
        scan_list = [scans_gos[0]] * 2 + [scans_gos[1]] * 2
        band_list = 2 * bands_gos
        for ii, block in enumerate(blocks):
            block.verify()
            the_scan = block.field['SCAN']
            the_band = round(block.field['CRVAL1'][0], -4)
            self.assertEqual(scan_list[ii], the_scan)
            self.assertAlmostEqual(band_list[ii], the_band)


class TestWriter(unittest.TestCase) :
    """Unit tests for fits file writer.
    """

    def setUp(self) :
        self.writer = fits.SpecWriter()
        self.reader = fits.SpecReader(testfile_gos)
        block, = self.reader.read(0, 0)
        self.writer.add_data(block)

    def test_add_data(self) :
        for field_name in fields_gos:
            self.assertEqual(len(self.writer.field[field_name]),
                             ntime_gos*npol_gos*ncal_gos)
        block, = self.reader.read(1, 0)
        self.writer.add_data(block)
        for field_name in fields_gos:
            self.assertEqual(len(self.writer.field[field_name]),
                             2*ntime_gos*npol_gos*ncal_gos)

    def test_error_on_bad_format(self) :
        block, = self.reader.read(1, 0)
        block.field['CRVAL1']._fits_format = '1I'
        self.assertRaises(fits.DataError, self.writer.add_data, block)

    def tearDown(self) :
        del self.writer
        del self.reader


class TestCircle(unittest.TestCase) :
    """Circle tests for the reader and writer.

    I'm sure there is a word for it, but I've dubbed a circle test when you
    read some data, do something to it, then write it and read it again.  Then
    check it element by element that it hasn't changed.
    """

    def setUp(self) :
        self.reader = fits.SpecReader(testfile_gos)
        self.blocks = list(self.reader.read([], []))

    def circle(self) :
        self.blocksToWrite = self.blocks
        self.writer = fits.SpecWriter(self.blocksToWrite)
        self.writer.write('temp_test.fits')
        self.newReader = fits.SpecReader('temp_test.fits')
        self.newblocks = self.newReader.read()

        self.assertEqual(len(self.blocks), len(self.newblocks))
        for ii in range(len(self.newblocks)) :
            OldDB = self.blocks[ii]
            NewDB = self.newblocks[ii]
            self.assertEqual(OldDB.shape, NewDB.shape)
            self.assertTrue(ma.allclose(OldDB.data, NewDB.data))
            for field in fields_gos:
                self.assertEqual(OldDB.field[field].axes,
                                 NewDB.field[field].axes)
            for field in ['SCAN', 'CRPIX1', 'CDELT1'] :
                self.assertEqual(OldDB.field[field], NewDB.field[field])
            for field in ['OBJECT', 'TIMESTAMP', 'OBSERVER',]:
                self.assertEqual(str(OldDB.field[field]).strip(),
                                 str(NewDB.field[field]).strip())
            for field in ['BANDWID', 'RESTFREQ', 'DURATION'] :
                self.assertAlmostEqual(OldDB.field[field], NewDB.field[field])
            for field in ['CRVAL1', 'LST', 'ELEVATIO', 'AZIMUTH',
                          'CRVAL2', 'CRVAL3', 'EXPOSURE'] :
                self.assertTrue(np.allclose(OldDB.field[field], 
                                            NewDB.field[field]))
            for field in ['DATE-OBS'] :
                self.assertTrue(np.alltrue(np.equal(OldDB.field[field], 
                                            NewDB.field[field])))
            for field in ['CRVAL4', 'CAL'] :
                self.assertTrue(all(OldDB.field[field] == NewDB.field[field]))

    def test_basic(self) :
        self.circle()

    def test_masking(self) :
        self.blocks[1].data[3,2,30] = ma.masked
        self.circle()
        self.assertTrue(np.all(self.blocks[1].data.mask == 
                            self.newblocks[1].data.mask))

    def tearDown(self) :
        del self.reader
        del self.writer
        del self.newReader
        os.remove('temp_test.fits')



if __name__ == '__main__' :
    unittest.main()
