"""Bruker-to-ISMRMRD conversion library.

Convert Bruker MRI raw data (fid / rawdata) to ISMRMRD format.

Example
-------
>>> from bruker_to_ismrmrd import MrdData, convert
>>> mrd = convert('path/to/experiment')
>>> mrd = MrdData.from_bruker('path/to/experiment')  # equivalent
>>> mrd.save('output.mrd')
"""

from bruker_to_ismrmrd.converter import convert
from bruker_to_ismrmrd.mrd_data import MrdData

__all__ = ['MrdData', 'convert']
