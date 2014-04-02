"""Unit tests for :mod:`fits` module."""

import unittest
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
nfreq_gos = 2048


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




if __name__ == '__main__' :
    unittest.main()
