"""Python interface to ScannerBit library, from GAMBIT project"""

# Need to do a little voodoo so that dynamically loaded shared libraries
# can dynamically load other shared libraries (the scanner plugins)
import os
import sys
import ctypes
flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

from functools import partial
import inspect
import copy
import h5py

# Just doing this will initialise MPI, and it will automatically
# call 'finalize' upon exit. So we need do nothing except import this.
from mpi4py import MPI
MPI_rank = MPI.COMM_WORLD.Get_rank()
MPI_size = MPI.COMM_WORLD.Get_size()

# DEBUG! Only used to check that YAML options looks reasonable
import yaml

# Need to tell ScannerBit where its config files are located
# We do this via a special environment variable
gambit_path = os.path.dirname(__file__)
print("Setting GAMBIT_RUN_DIR to:",gambit_path)
os.environ["GAMBIT_RUN_DIR"] = gambit_path

# Other python helper tools
from .defaults import _default_options
from .utils import _merge
from .hdf5_help import get_data, HDF5 
from .processify import processify

class SanityCheckException(Exception):
    """Exception thrown by sanity checks of user-supplied input"""
    pass

def _add_default_options(options):
   """Inspect user-supplied options, and fill in any missing
      bits with defaults"""
   _merge(options,_default_options)
   return options

# Better version of 'partial'. From 'funcy' library: https://github.com/Suor/funcy
def func_partial(func, *args, **kwargs):
    """A functools.partial alternative, which returns a real function.
       Can be used to construct methods."""
    return lambda *a, **kw: func(*(args + a), **dict(kwargs, **kw))

class SafeVec:
   """A simple wrapper class for the 'vec' argument to user prior functions,
      with nicer error messages"""
   def __init__(self,vec,size):
       self.vec = vec
       self.size = size

   def __getitem__(self, key):
       try:
           if float(key).is_integer():
             if key >= self.size:
                msg="Error accessing item '{0}' in 'vec' argument of user-supplied prior function! The size of 'vec' is {1} so entry {0} doesn't exist!".format(key,self.size) 
                raise SanityCheckException(msg)
             elif key < 0:
                msg="Error accessing item '{0}' in 'vec' argument of user-supplied prior function! The 'vec' behaves like a list and must be indexed by a non-negative integer.".format(key) 
                raise SanityCheckException(msg)
           else:
             msg="Error accessing item '{0}' in 'vec' argument of user-supplied prior function! 'vec' behaves like a list and must indexed by an integer.".format(key)
             raise SanityCheckException(msg)
       except (ValueError, AttributeError):
          msg="Error accessing item '{0}' in 'vec' argument of user-supplied prior function! '{0}' does not seem to be an integer, or even a number!".format(key)
          raise SanityCheckException(msg)
       return self.vec[key]

   def __setitem__(self, key, value):
       msg="Error setting item '{0}' to '{1}' in 'vec' argument of user-supplied prior function! 'vec' is read-only!".format(key,value) 
       raise SanityCheckException(msg)

@processify # Run this function in a separate process, so that scanner plugins can be re-loaded between scans
def _run_scan(settings, loglike_func, prior_func):
   """Perform a scan. This function is decorated in such a 
      way that it runs in a new process. This is important
      because the GAMBIT plugins can only run once per
      process, because the shared libraries need to be reloaded
      to perform a second scan.
      """
   # Import functions from the ScannerBit.so library
   # Should also trigger the loading of the scanner plugin libraries
   from .ScannerBit.python import ScannerBit

   # Check that ScannerBit was compiled with MPI enabled if we are using more than one process
   if MPI_size>1 and not ScannerBit.WITH_MPI:
       msg = "ScannerBit has not been compiled with MPI enabled! Please try again using only one process."
       raise RuntimeError(msg)

   print("prior_func:",prior_func)
   print("prior_func:",inspect.signature(prior_func))

   # Attach the ScannerBit object to the first argument of the wrapped likelihood and prior functions
   wrapped_loglike = func_partial(loglike_func,ScannerBit)
   if prior_func is not None:
       wrapped_prior = func_partial(prior_func,ScannerBit)
   else:
       wrapped_prior = None

   # Create scan object
   myscan = ScannerBit.scan(False) # Do not Init_MPI in ScannerBit, we already did it out here in the wrapper.

   # Double check settings fed to ScannerBit!
   print("Scan settings:")
   print(settings)
   print("==============")
   print(yaml.dump(settings, default_flow_style=False))

   # run scan
   # 'inifile' can be the name of a YAML file, or a dict.
   #myscan.run(inifile=settings, lnlike={"LogLike": wrapped_loglike}, prior=prior_func, restart=True)
   ret = myscan.run(inifile=settings, lnlike={"LogLike": wrapped_loglike}, prior=wrapped_prior, restart=True)
   
   if ret!=0:
      msg = "Fatal error encountered while running ScannerBit!"
      raise RuntimeError(msg)

   MPI.COMM_WORLD.Barrier()
   rank = MPI.COMM_WORLD.Get_rank()
   print("Rank {0} passed scan end barrier!".format(rank))

class Scan:
    """Helper object for setting up and running a scan, and
       making some basic plots.
    """
    def __init__(self, function, prior_func=None, bounds=None, prior_types=None, kwargs=None, scanner=None,
      scanner_options={}, model_name=None, output_path=None, fargs=None):
        """
        function - Python function to be scanned
        prior_func - User-define prior transformation function (optional)
        bounds - list of ranges of parameter values (function arguments) to scan,
                 or mean/std-dev in case of normal prior.
        prior_types - list of priors to use for scanning in each dimension (e.g. flat/log).
                      If None then the user is expected to do the inverse transform from
                      a unit hypercube to their parameter space themselves.
        scanner - Scanning algorithm to use
        scanner_options - Configuration options dictionary for chosen scanning algorithm
        f_args - List of names of function argments to use (if None these are
         inferred from the signature of 'function')
        """
        self.function = function
        self.prior_func = prior_func

        # Determine parameter names, either automatically or from 'fargs' argument
        if fargs is None:
            signature = inspect.signature(self.function)
            print("signature:", signature.parameters.items())
            # First argument needs to be the 'scan' object, to allow access to printers etc. Skip it in determination of parameter names.
            par_names = signature.parameters.keys()
            self._argument_names = list(par_names)[1:]
            if list(par_names)[0] is not "scan":
                msg="\n\nError in signature of user-supplied log-likelihood function! The first argument must be named 'scan' and will be passed a reference to a global ScannerBit object that can be used to e.g. print extra information to the scan output file.\n"
                raise SanityCheckException(msg)
        else:
            self._argument_names = fargs

        # Determine prior to be used
        self.bounds = bounds if bounds else [(0,1)] * len(self._argument_names)
        self.prior_types = prior_types if prior_types else ["flat"] * len(self._argument_names)
        self.scanner = scanner

        if scanner not in _default_options["Scanner"]["scanners"].keys():
            msg = "Unknown scanner '{0}' was selected! Please choose from the following available scanning algorithms:".format(scanner)
            for s in _default_options["Scanner"]["scanners"].keys():
                msg += "\n   {0}".format(s)
            raise ValueError(msg)

        if MPI_size>1 and scanner in ["random","toy_mcmc"]:
            msg = "Scanner {0} selected, however MPI_size>1, and unfortunately this algorithm is not yet parallelised. Please either choose another sampling algorithm, or run this algorithm on one process only."
            raise ValueError(msg)

        # Copy user-supplied scanner options into full scan settings dictionary
        self.settings = _add_default_options(copy.deepcopy({"Scanner": {"scanners": {scanner: scanner_options}}}))
        self.kwargs = kwargs

        print("self._argument_names:", self._argument_names)
        print("self.bounds:", self.bounds)
        assert len(self._argument_names) == len(self.bounds)
 
        if model_name is None:
           if "Parameters" in self.settings:
              #print(self.settings["Parameters"])
              self._model_name = list(self.settings["Parameters"].keys())[0]
           else:
              self._model_name = "default"
        else:
           self._model_name = model_name

        # Wrap user-supplied likelihood and prior functions with some extra sanity checking
        self._wrapped_function = self._wrap_function()
        if prior_func is not None:
           self._wrapped_prior = self._wrap_prior()
        else:
           self._wrapped_prior = None

        self._scanned = False

        # Make up a name for the run, if user didn't provide one
        if output_path is None:
            self.settings["KeyValues"]["default_output_path"] += "{0}_scan".format(self.scanner)
        else:
            self.settings["KeyValues"]["default_output_path"] = output_path

        self._process_settings()
        self.loglike_par = "LogLike"
        self.scanner = self.settings["Scanner"]["use_scanner"] # Might have been None and then filled by _process_settings with defaults
        if self.scanner == "multinest" \
        or self.scanner == "polychord":
            self.posterior_par = "Posterior"
        else:
            self.posterior_par = None

    def _get_hdf5_group(self):
        """
        """
        assert self.settings["Printer"]["printer"] == "hdf5"
        file_name = self.settings["Printer"]["options"]["output_file"]
        group_name = self.settings["Printer"]["options"]["group"]
        DIR = self.settings["KeyValues"]["default_output_path"]
        fullpath = "{}/samples/{}".format(DIR, file_name)
        f = h5py.File(fullpath,'r')
        g = f[group_name]
        return g

    def _process_settings(self):
        """Copy 'pythonic' options into the settings dictionary
           so that they can be automaticall converted into YAML 
           by the API, which ScannerBit can then read"""
        if(self.scanner is not None):
            self.settings["Scanner"]["use_scanner"] = self.scanner
        if("Parameters" not in self.settings):
            self.settings["Parameters"] = dict()
            self.settings["Parameters"][self._model_name] = dict()
            for n in self._argument_names:
                # Just add parameter names. We will write the prior settings out fully
                # in the Priors section next
                self.settings["Parameters"][self._model_name][n] = None
        if("Priors" not in self.settings):
            self.settings["Priors"] = dict()
            if (self.prior_types is not None) \
               and (self.bounds is not None):
                for n, b, t in zip(self._argument_names, self.bounds, self.prior_types):
                    prior_setup = {'prior_type': t, 'parameters': ["{0}::{1}".format(self._model_name,n)]}
                    if t is 'gaussian' or t is 'cauchy':
                       prior_setup['mean'] = [b[0]]
                       prior_setup['cov']  = [b[1]] # Assume this is the VARIANCE, not standard deviation
                    else:
                       prior_setup['range'] = b
                    self.settings["Priors"]["{0}_prior".format(n)] = prior_setup
            else:
                raise ValueError("No prior settings found! These need to be either supplied in simplified form via the 'bounds' and 'prior_types' arguments, or else supplied in long form (following the GAMBIT YAML format) in the 'settings' dictionary under the 'Priors' key (or under the 'Parameters' key in the short-cut format)")
        #print(self.settings)
        print("Scan settings in:")
        print("==============")
        print(yaml.dump(self.settings, default_flow_style=False))

    def _wrap_function(self):
        """
        """
        def wrapped_function(scan,par_dict):
            """
            """
            arguments = [par_dict["{0}::{1}".format(self._model_name, n)]
              for n in self._argument_names]
            return self.function(scan, *arguments, **(self.kwargs or {}))

        return wrapped_function

    def _wrap_prior(self):
       """Wrap the user-supplied prior transformation function so that we can
          do some sanity checking on it"""
       def wrapped_prior(scan,vec,map):
          #Tell ScannerBit the dimension of the parameter space
          size=len(self._argument_names)
          scan.ensure_size(vec,size)
          tmp_map = {}
          self.prior_func(SafeVec(vec,size),tmp_map)
          # Check that user added all the parameters
          has_model_name=None
          for p in self._argument_names:
             if self._model_name+"::"+p not in tmp_map.keys() \
               and p in tmp_map.keys():
                if has_model_name is None: 
                   has_model_name=False
                elif has_model_name is True:
                   msg="Error in user-supplied prior transformation function! Parameter names set in 'map' are inconsistent; some appear to have the model name prefix, while others do not. Please edit your prior function to use only one format or the other."  
                   raise SanityCheckException(msg)
             elif p not in tmp_map.keys():
                   msg="Error in user-supplied prior transformation function! User must define parameters {0} or {1} in the 'map' argument, however parameter {2} was not found! Please fix your prior transformation function.".format(self._argument_names,[self._model_name+"::"+x for x in self._argument_names],p)
                   raise SanityCheckException(msg)
             else:
                if has_model_name is None: 
                   has_model_name=True
                elif has_model_name is False:
                   msg="Error in user-supplied prior transformation function! Parameter names set in 'map' are inconsistent; some appear to have the model name prefix, while others do not. Please edit your prior function to use only one format or the other."
                   raise SanityCheckException(msg)
          if not has_model_name:
             # If they added them without the 'model' part of the name then add that automatically now
             for k,v in tmp_map.items():
                 map[self._model_name+"::"+k] = v
          else:
             for k,v in tmp_map.items():
                 map[k] = v
       return wrapped_prior

    def scan(self):
       """Perform a scan. This runs a function that is decorated in such a 
       way that it runs in a new process. This is important
       because the GAMBIT plugins can only run once per
       process, because the shared libraries need to be reloaded
       to perform a second scan.

       Downside is that all arguments must be pickle-able.
       """
       try:
           _run_scan(self.settings, self._wrapped_function, self._wrapped_prior)
       except RuntimeError as err:
           # Error messages thrown in the subprocess get kind of mangled, need to
           # help it print correctly
           print("Error thrown from ScannerBit subprocess:\n")
           print(err.args[0].replace('\\n', '\n')) # Newlines appear to be weirdly escaped. Fix them. 
           quit()

       self._scanned = True

    def get_hdf5(self):
        try:
            g = HDF5(self._get_hdf5_group().id,model=self._model_name,
                loglike=self.loglike_par, posterior=self.posterior_par)
        except:
            if(self._scanned):
                raise IOError("Failed to open HDF5 output of scan!")
            else:
                raise IOError("Failed to open HDF5 output of scan, however we did not perform a scan just now. The output will only exist if you have previously run this scan. Please check that you did this!")
        return g
