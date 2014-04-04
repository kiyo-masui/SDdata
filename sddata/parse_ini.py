"""
Python parameter file parser.

Framework for writing and parsing input files for large programs.  The input
file will have plain python syntax.

Functions
=========

.. autosummary::
   :toctree: generated/

    parse
    write_params

Exceptions
==========

.. autosummary::
   :toctree: generated/

    ParameterTypeError
    ParameterFileError

"""


import logging

logger = logging.getLogger(__name__)


def parse(ini_data, params, return_undeclared=False, prefix='',
          type_check=False):
    """Parses a python file or dictionary to get parameters.
    
    This function accepts a filename and a dictionary of keys and pre-typed
    values. It returns a dictionary of the same keys with values read from
    file.  It optionally performs type checking.

    Parameters
    ----------
    ini_data : string filename or dictionary
        The file must contain a script that defines parameter values in the
        local namespace.  Alternately, if a dictionary, then parameters are
        read from the dictionary.  Variables must have names and types
        corresponding to the params dictionary argument.
    params : dict
        Keys and correspond to variable names to be read and values correspond
        to defaults if the corresponding variable is not found in the file.
    return_undeclared : bool
        Whether to return a second dictionary of with variables found in the
        parameter file but not in the in params argument.
    prefix : string
        A prefix added to parameter names (defined in the keys of params) when
        read from the input file or dictionary.  The prefix is not added to the
        returned output dictionary.
    type_check : bool
        Whethar to raise an exception if the recived value for a parameter is a
        different type than the default value.

    Returns
    -------
    out_params : dict
        Keys are the same as *params* but with values read from file.
    undeclared: dict, optional
        A dictionary that holds any key found in the file but
        not in params. Returned if *return_undeclared* is true.

    """
    
    if isinstance(ini_data, str):
        logger.info('Reading parameters from file: '+ ini_data)
        # Convert local variables defined in python script to dictionary.
        # This is in a separate function to avoid namespace issues.
        dict_to_parse = _execute_parameter_file(ini_data)
    elif isinstance(ini_data, dict):
        logger.info('Reading parameters from dictionary.')
        dict_to_parse = ini_data
    elif ini_data is None :
        logger.info('No input, all parameters defaulted.')
        if return_undeclared:
            return dict(params), {}
        else:
            return dict(params)
    else:
        raise TypeError("Argument ini must be a dictionary, file name, "
                        "or None (to accept defaults).")
    
    return parse_dict(dict_to_parse, params, return_undeclared, prefix,
                      type_check)


def parse_dict(dict_to_parse, params, return_undeclared=False, prefix='',
               type_check=False):
    """Same as parse_ini.parse except parameters read from only dictionary.
    
    This function is intended for internal use.  All of it's functionality is
    availble from the parse function.

    This function accepts an input dictionary and a dictionary of keys 
    and pre typed
    values. It returns a dictionary of the same keys with values read from
    the input dictionary.  See the docstring for parse for more
    information, the only difference is the first argument must be a
    dictionary.

    Parameters
    ----------
    dict_to_parse : dict
        A dictionary containing keys and values to be read as parameters.
        Entries should have keys and types corresponding to the pars dictionary
        argument (depending on level of checking requested).

    """
    
    # Same keys as params but for checking but contains only a flag to indicate
    # if parameter retained it's default value.
    defaulted_params = {}
    for key in params.iterkeys():
        defaulted_params[key] = True
    # Make dictionaries for outputs
    undeclared = {}   # For keys found in dict_to_parse and not in params
    out_params = dict(params)

    # Loop over both input dictionaries and look for matching keys
    for inkey, invalue in dict_to_parse.iteritems():
        found_match_flag = False
        for key, value in params.iteritems():
            # Check for matching keys. Note stripping.
            if prefix + key.strip() == inkey.strip():
                if type(value) != type(invalue):
                    if type_check:
                        raise ParameterTypeError(
                            "Tried to assign an input "
                            "parameter to the value of the wrong type " 
                            "and asked for strict type checking. "
                            "Parameter name: " + key)
                    logger.warning("Assigned an input "
                            "parameter to the value of the wrong type. "
                            "Parameter name: " + key)
                out_params[key] = invalue
                found_match_flag = True
                defaulted_params[key]=False
                # There shouldn't be another matching key so:
                break
        if not found_match_flag :
            # Value found in dict_to_parse was not found in params
            undeclared[inkey]=invalue
    # Check if parameters have remained a default value and print information
    # about the parameters that were set. Depending on feedback level.
    logger.info("Parameters set.")
    for key, value in out_params.iteritems():
        if defaulted_params[key] :
            logger.info("parameter: "+key+" defaulted to value: " 
                    + str(value))
        else:
            logger.debug("parameter: "+key+" obtained value: "
                    + str(value))

    if return_undeclared :
        return out_params, undeclared
    else :
        return out_params


def _execute_parameter_file(this_parameter_file_name):
    """
    Executes python script in named file and returns dictionary of variables
    declared in that file.
    """
    
    # Only a few locally defined variables and all have a long name to avoid
    # namespace conflicts.

    # Execute the filename which presumably holds a python script. This will
    # bring the parameters defined there into the local scope.
    try:
        exec(open(this_parameter_file_name).read())
    except Exception as E:
        msg = ("Execution of parameter file " + this_parameter_file_name +
               " caused an error.  The error message was: " + repr(E))
        raise ParameterFileError(msg)
    # Store the local scope as a dictionary.
    out = locals()
    # Delete all entries of out that correspond to variables defined in this
    # function (i.e. not in the read file).
    del out['this_parameter_file_name']
    # Return the dictionary of parameters read from file.
    return out


def write_params(params, file_name, prefix='', mode='w') :
    """Write a parameter dictionary to file.

    Given a dictionary of parameters, such as one of the ones read br the parse
    function, this program writes a compatible ini file.
    
    This should work if the parameters are built in types, but no promises for
    other types. Basically if the out put of 'print param' looks like it could
    go on the rhs of the assignment operator, you are in good shape.

    Parameters
    ----------
    params : dict
        Parameter names and values to be written to file.
    file_name: sting
        File to write to.
    prefix : string
        prefix for the parameter names when written to file.
    mode: stirng
        Valid values are 'a' or 'w'.  Whether to open the file in write or
        append mode.

    """
    
    if not (mode == 'w' or mode == 'a'):
        raise ValueError("Params can be written with mode either 'w' or 'a'.")
    file = open(file_name, mode)
    for par_name, value in params.iteritems():
        line_str = prefix + par_name + ' = '
        try :
            line_str = line_str + repr(value)
        except SyntaxError:
            line_str = line_str + repr(value)
        line_str = line_str + '\n'
        file.write(line_str)
    file.close()


# Exceptions
# ----------

class ParameterTypeError(Exception):
    """Raised when a parameter is not the expected type."""


class ParameterFileError(Exception):
    """Raise when there is an error executing a parameter file."""
