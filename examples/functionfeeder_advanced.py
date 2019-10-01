"""
This script shows how the FunctionFeeder class can be used to create custom
sets of TestFunctions, each with their own configuration.
"""
from high_dimensional_sampling import functions

feeder = functions.FunctionFeeder()
feeder.load_function("Beale")
feeder.load_function("Block", {'dimensionality': 3})
feeder.load_function("Block", {'dimensionality': 4})
feeder.load_function("Block", {'dimensionality': 4, 'block_size': 2})
feeder.load_function("Easom")
feeder.load_function("Easom", {'absolute_range': 40})
feeder.load_function("Easom", {'absolute_range': 32, 'name': 'Easom_specific'})

# Show the names of the functions (see that duplicates exist)
print("TestFunction names prior to duplicate removal:")
for f in feeder.functions:
  print("- {}".format(f.name))

# Correct duplicates
feeder.fix_duplicate_names()

# Show names again
print("\nTestFunction names after duplicate removal:")
for f in feeder.functions:
  print("- {}".format(f.name))
