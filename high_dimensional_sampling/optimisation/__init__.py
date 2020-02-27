from .randomoptimisation import RandomOptimisation  # noqa: F401
from .cmaesoptimisation import CMAOptimisation  # noqa: F401

""" The `pyscannerbit` implementation is excluded from unit tests, as we
-- unfortunately -- did not get pyscannerbit to work within Travis """
from .pyscannerbit import PyScannerBit  # noqa: F401
