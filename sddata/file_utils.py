"""
Short utilities for files and file paths.

Most of these are only a few lines long, but are quite useful.

Functions
=========

.. autosummary::
   :toctree: generated/

    mkdir_p
    mkparents
    abbreviate_file_path


"""

import os 
import errno

def mkdir_p(path) :
    """Same functionality as shell command shell command mkdir -p."""
    try:
        os.makedirs(path)
    except OSError, exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise

def mkparents(path) :
    """Given a file name, makes all the parent directories."""

    last_slash = path.rfind('/')
    if last_slash > 0 : # Protect against last_slash == -1 or 0.
        outdir = path[:last_slash]
        mkdir_p(outdir)

def abbreviate_file_path(fname) :
    """Abbreviates file paths.
    
    Given any file path return an abbreviated path showing only the deepest
    most directory and the file name (Useful for writing feedback that doesn't
    flood your screen).
    """
    
    split_fname = fname.split('/')
    if len(split_fname) > 1 :
        fname_abbr = split_fname[-2] + '/' + split_fname[-1]
    elif len(split_fname) > 2 :
        fname_abbr = (split_fname[-3] + '/' + split_fname[-2]
                      + '/' + split_fname[-1])
    else :
        fname_abbr = fname

    return fname_abbr
