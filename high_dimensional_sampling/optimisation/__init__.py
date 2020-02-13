from .randomoptimisation import RandomOptimisation  # noqa: F401
from .particlefilter import ParticleFilter  # noqa: F401
from .gpyopt import GPyOpt  # noqa: F401
from .cmaesoptimisation import CMAOptimisation  # noqa: F401
from .ampgo import Ampgo  # noqa: F401
from .pygmo import Pygmo  # noqa: F401

""" The `pyscannerbit` implementation is excluded from unit tests, as we
-- unfortunately -- did not get pyscannerbit to work within Travis """
from .pyscannerbit import PyScannerBit  # noqa: F401
