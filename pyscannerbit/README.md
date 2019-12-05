pyScannerBit: A Python wrapper for the ScannerBit module of GAMBIT
===

For full docs please visit our [readthedocs](https://pyscannerbit.readthedocs.io) pages (note, still under development).

Dependencies
---

This package requires the following:
- Python 3
- HDF5 and h5py
- C++11 compatible compilers
- a POSIX operating system (no Windows, sorry)

The following may optionally be provided by your system, but if they aren't
then this package will use internal versions of them
- yaml-cpp
- pybind11


Installation
---

### For developers, via git repository

In order to make sure everything will build properly when the installing
the package via PyPI repositories, it is best to do local test installations as
follows:
 
    python setup.py sdist
    pip install dist/pyscannerbit-0.0.1.tar.gz  

This creates the source distribution tarball that is uploaded to PyPI, and
performs the installation directly from that. Following this procedure means
that nothing will accidentally get left out of the distribution tarball. If
you need to add new files to it, add them to the `MANIFEST.in` file.

### For users

If you have downloaded the source from github then feel free to use
the developer installation instructions above. But you can also install this
package via PyPI:

    pip install pyscannerbit


Package structure
---

This information is mainly for developers, but it might be helpful to know this
if you are having installation trouble (though if this is the case please file
or check the bug reports at `https://github.com/bjfar/pyscannerbit/issues`)

This package is primarily a Python interface to a C API that is built by the
GAMBIT build system. This system in turn automatically downloads and build
various scanning algorithm libraries and turns them into plugins for ScannerBit.

The tricky thing that needs to occur, which does not occur in a 'vanilla'
GAMBIT installation, is that various shared libraries need to be moved into
the final Python package installation path, whilst still being able to
find each other when Python loads them. This involves some careful handling
of their installation and rpath settings. If you are getting errors about
missing symbols when trying to `import pyscannerbit`, the reasons is probably
that something has gone wrong with this process. It is still a bit experimental,
so please file a bug report if you see such errors.

GAMBIT-side modifications
---

If you modify anything on the GAMBIT side, those changes will need to be
manually imported into this package. This is done via the ScannerBit
standalone tarball. So in the GAMBIT source do

    cmake ..
    make standalone_tarballs

then from this package run the script `grab_ScannerBit.sh` as follows

    ./grab_ScannerBit.sh <GAMBIT_SOURCE_ROOT>

and it will strip out contrib packages that we don't need, and then copy
the stripped tarball into pyscannerbit/scannerbit.

NOTE! The GAMBIT version number is not automatically detected, so if
this changes it will need to be updated in the `grab_ScannerBit.sh` script.  

Upload new versions to PyPI
---

Mostly for my own sake because I keep forgetting how to do this:

    twine upload dist/pyscannerbit-x.x.x.tar.gz 

Known issues
---

License
---
Copyright (c) 2018, The GAMBIT Collaboration
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

