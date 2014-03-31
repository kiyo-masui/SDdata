"""Unit tests for :mod:`fits` module."""

import unittest

import numpy as np
import numpy.ma as ma

import sddata.fits as fits

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


if __name__ == '__main__' :
    unittest.main()
