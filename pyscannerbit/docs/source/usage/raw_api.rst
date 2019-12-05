.. _raw api:

Low-level API
============

For user-friendliness we recommend using the high-level API described in :ref:`quick start` and most of the rest of these docs. However, if for some reason you don't want to use this API, then you have the option of digging deeper, into a lower level API that is exposed by ScannerBit. Below is a short description of how this API can be used.

First, you need to set some flags for :code:`dlopen` to help than scanner plugin libraries be dynamically loaded correctly (this is done automatically in the high-level API)::

    import sys
    import ctypes
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

Next import the 'bare-bones' interface from a few levels down in the package::

    # Dig past the extra python wrapping to the direct ScannerBit.so shared library interface level
    from pyscannerbit.ScannerBit.python import ScannerBit
 
Define the log-likelihood function you wish to scan, and (if desired) a prior transformation function::

    def loglike(m):
        a = m["model1::x"]
        ScannerBit.print("my_param", 0.5) # Send extra data to the output file at each point 
        return -a*a/2.0

    def prior(vec, map):
        # tell ScannerBit that the hypergrid dimension is 1
        ScannerBit.ensure_size(vec, 1) # this needs to be the first line!
        map["model1::x"] = 5.0 - 10.0*vec[0]

These look superficially similar to the functions that should be supplied to the high-level API, however please note that there is no sanity/error checking in this low-level API, so mistakes will result in cryptic errors from inside ScannerBit. The model name in e.g. :code:`model1::x` is also non-optional in this interface, and similarly you must call the :code:`ensure_size` function in the prior function to tell ScannerBit the dimension of your parameter space.

Next generate a settings dictionary. The structure of this should match the YAML format required by ScannerBit when run via GAMBIT (this is one thing that we will simplify in the nice wrapper....)::

    settings = {
    "Parameters": {
      "model1": {
        "x": None,
        }
      },
    "Priors": {
      "x_prior": {
        "prior_type": 'flat',
        "parameters": ['model1::x'],
        "range": [1.0, 40.0],
        }
      },
    "Printer": {
      "printer": "hdf5",
      "options": {
        "output_file": "results.hdf5",
        "group": "/",
        "delete_file_on_restart": "true",
        }
      },
    "Scanner": {
      "scanners": {
        "twalk": {
          "plugin": "twalk",
          "like": "LogLike",
          "tolerance": 1.003,
          "kwalk_ratio": 0.9,
          "projection_dimension": 4
          }
        },
      },
    "KeyValues": {
      "default_output_path": "pyscannerbit_run_data/",
      "likelihood": {
        "model_invalid_for_lnlike_below": -1e6
        }
      } 
    }
    
Run your scan!::

    myscan = ScannerBit.scan(True)
    myscan.run(inifile=settings, lnlike={"LogLike": like}, prior=prior, restart=True)

If all went well, your scan should begin, and generate HDF5 format output in :code:`pyscannerbit_run_data/samples/results.hdf5`.
