import import_yaml

template = """
Parameters:
  python_model: None

Priors:

Printer:

  printer: hdf5
  options:
    output_file: "results.hdf5"
    group: "/python"
    delete_file_on_restart: true

Scanner:

  use_scanner: None

  scanners:

    de:
      plugin: diver
      like: LogLike
      NP: 1000

    multinest:
      plugin: multinest
      like:  LogLike
      nlive: 1000
      tol: 0.1

    mcmc:
      plugin: great
      like: LogLike
      nTrialLists: 5
      nTrials: 10000

    twalk:
      plugin: twalk

Logger:
  redirection:
    [Default]      : "default.log"
    [Scanner]      : "Scanner.log"

KeyValues:
  likelihood:
    model_invalid_for_lnlike_below: -1e6
"""

settings = import_yaml.load(template)
