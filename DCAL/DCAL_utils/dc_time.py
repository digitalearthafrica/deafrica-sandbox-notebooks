import numpy as np
from datetime import datetime

def dt_to_str(date, fmt='%Y-%m-%d'):
    """
    Converts a datetime object to a string.
    """
    return date.strftime(fmt)

def _n64_to_datetime(n64):
    """
    Converts Numpy 64 bit timestamps to datetime objects. Units in seconds
    """
    return datetime.utcfromtimestamp(n64.tolist() / 1e9)

def _n64_datetime_to_scalar(dt64):
    """
    Converts a NumPy datetime64 object to the number of seconds since 
    midnight, January 1, 1970, as a NumPy float64.
    
    Returns
    -------
    scalar: numpy.float64
        The number of seconds since midnight, January 1, 1970, as a NumPy float64.
    """
    return (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

def _scalar_to_n64_datetime(scalar):
    """
    Converts a floating point number to a NumPy datetime64 object.
    
    Returns
    -------
    dt64: numpy.datetime64
        The NumPy datetime64 object representing the datetime of the scalar argument.
    """
    return (scalar * np.timedelta64(1, 's')) + np.datetime64('1970-01-01T00:00:00Z')