.. _installation:

Installation
==================

.. _GAMBIT: https://gambit.hepforge.org/
.. _ScannerBit: https://arxiv.org/abs/1705.07959
.. _github: https://github.com/bjfar/pyscannerbit
.. _anaconda: https://conda.io/docs/user-guide/install/download.html
.. _pip: https://pip.pypa.io/en/stable/installing/

PyPI installation::

    pip install pyscannerbit

From `github`_::

    git clone https://github.com/bjfar/pyscannerbit.git
    pip install ./pyscannerbit

Either way it is recommended to use `pip`_ to install the package since this should generally be compatible with `anaconda`_ environments. ScannerBit will built from source, so don't worry if it takes a couple of minutes. If there is an error during the build then there is a good chance that some non-python dependency was missing, so check out the CMake output and see if you can figure out what that might have been. Feel free to ask for help on our `github`_ page!
