_default_options = {
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
    "diver": {
      "plugin": "diver",
      "like": "LogLike",
      "verbosity": 2,
      "NP": 1000,
      },
    "multinest": {
      "plugin": "multinest",
      "like":  "LogLike",
      "nlive": 1000,
      "tol": 0.5,
      },
    "polychord": {
      "plugin": "polychord",
      "like":  "LogLike",
      "nlive": 500,
      "tol": 0.5,
      },
    "twalk": {
      "plugin": "twalk",
      "sqrtR": 1.01, # This is a convergence criterion
      },
    "random": {
      "plugin": "random",
      "point_number": 10000,
      "like": "LogLike",
      },
    "toy_mcmc": {
      "like": "LogLike",
      "plugin": "toy_mcmc",
      "point_number": 10000,
      },
    "badass": {
      "like": "LogLike",
      "plugin": "badass",
      "points": 1000, 
      "jumps": 10,
      },
    "pso": {
      "plugin": "jswarm",
      "like":  "LogLike",
      "NP": 400,
      "adaptive_phi": "true",
      "adaptive_omega": "true",
      "verbosity": 2,
      },
    },
  },
"KeyValues": {
  "default_output_path": "pyscannerbit_run_data/",
  "likelihood": {
    "model_invalid_for_lnlike_below": -1e6
    }
  }
}

