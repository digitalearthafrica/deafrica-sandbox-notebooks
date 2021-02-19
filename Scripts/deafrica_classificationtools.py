# Functions in this script have been moved to a new location to allow them to be imported into notebooks as a Python package: ../Tools/

import pathlib
import warnings

warnings.warn("Scripts/deafrica_* scripts have been deprecated in favour of the deafrica-tools module. Please import deafrica_tools.classificationtools instead.", DeprecationWarning)

current_dir = pathlib.Path(__file__).parent.absolute()

import sys
sys.path.insert(1, str(current_dir.parent.absolute() / 'Tools'))

from deafrica_tools.classificationtools import *