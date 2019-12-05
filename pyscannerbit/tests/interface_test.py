import sys
import ctypes
import inspect

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

# Dig past the extra python wrapping to the direct ScannerBit.so shared library interface level
import pyscannerbit.scan as sb
from pyscannerbit.ScannerBit.python import ScannerBit

# define likelihood, technically optional
def like(m):
    #print("m:", m)
    #print("m.items():", [(k,v) for k,v in m.items()])
    a = m["model1::x"]
    ScannerBit.print("my_param", 0.5) # can print custom parameters 

    return -a*a/2.0

def like_a(a,b):
    ScannerBit.print("a", a) # can print custom parameters
    ScannerBit.print("b", b) # can print custom parameters
    return -a*a/2.0

def like_x(x,y):
    ScannerBit.print("x", x) # can print custom parameters
    ScannerBit.print("y", y) # can print custom parameters
    return -x*y/2.0
     
# Try to dynamically generate likelihood function
def wrap_function():
    def wrapped_function(m):
        #print("m:", m)
        a = m["model1::x"]
        b = 3
        return like_a(a,b) 
    return wrapped_function

# Do it from a class
class Test:
    def __init__(self,f):
        self.f_in = f
        self.f = self._wrap_function()
        signature = inspect.getargspec(self.f_in)
        self._argument_names = signature.args
        self._model_name = "model1"
        print("self._argument_names:", self._argument_names)
 
    def _wrap_function(self):
        def wrapped_function(par_dict):
            print("par_dict:", par_dict)
            print("par_dict.items():", [(k,v) for k,v in par_dict.items()])
            arguments = [par_dict["{}::{}".format(self._model_name, n)]
              for n in self._argument_names]
            return self.f_in(*arguments, **(self.kwargs or {}))
        return wrapped_function


#like_dyn = wrap_function()
t = Test(like_x)
like_dyn = t.f 

print("like_dyn:",like_dyn)
print("args: ",inspect.getargspec(like_dyn))

# Version where ScannerBit interface is passed through via the wrapper
# (needed to allow multiple scans per python session)
def like2(scan,m):
    a = m["model1::x"]
    scan.print("my_param", 0.5) # can print custom parameters 
    return -a*a/2.0


# define prior, optional
def prior(vec, map):
    # tell ScannerBit that the hypergrid dimension is 1
    ScannerBit.ensure_size(vec, 1) # this needs to be the first line!

    map["model1::x"] = 5.0 - 10.0*vec[0]

# declare scan object
myscan = ScannerBit.scan(True)

settings = {
"Parameters": {
  "model1": {
    "x": None,
    "y": None,
    }
  },
"Priors": {
  "x_prior": {
    "prior_type": 'flat',
    "parameters": ['model1::x'],
    "range": [1.0, 40.0],
    },
  "y_prior": {
    "prior_type": 'flat',
    "parameters": ['model1::y'],
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
  "use_scanner": "twalk",
  },
"KeyValues": {
  "default_output_path": "pyscannerbit_run_data/",
  "likelihood": {
    "model_invalid_for_lnlike_below": -1e6
    }
  }
}

#myscan.run(inifile=settings, lnlike={"LogLike": like}, prior=prior, restart=True)

# Try via wrapper to the above interface:
#sb._run_scan(settings, like, prior)

# Try with dummy prior
def dummy_prior(vec, map):
    ScannerBit.ensure_size(vec, 2) # this needs to be the first line!
    map["model1::x"] = 5.0 - 10.0*vec[0]
    map["model1::y"] = 5.0 - 10.0*vec[1]

myscan.run(inifile=settings, lnlike={"LogLike": like}, prior=dummy_prior, restart=True)

# Try with dynamically created function
#sb._run_scan(settings, like_dyn, dummy_prior)

#Try without prior
#sb._run_scan(settings, like2, "") # Fails!
 
