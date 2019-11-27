"""
This example script shows how different __call__ interfaces of a TestFunction
can be accessed.
"""
import high_dimensional_sampling as hds
import numpy as np

# Create a test function
reciprocal = hds.functions.Reciprocal()

# Create input data for both variables and merge them
variable_1 = np.random.rand(5).reshape(-1, 1)
variable_2 = np.random.rand(5).reshape(-1, 1)
x = np.hstack((variable_1, variable_2))

""" Numpy interface """
# Evaluate testfunction with numpy interface
print("Numpy interface (normal)")
print(reciprocal(x))
print("\nNumpy interface (derivative)")
print(reciprocal(x, derivative=True))

""" Simple interface """
# Get simple interface
simple = reciprocal.get_simple_interface()

# Evaluate test function with simple interface
print("\nSimple interface (normal)")
print(simple(variable_1, variable_2))
print("\nSimple interface (derivative)")
print(simple(variable_1, variable_2, derivative=True))

# Note that the output of the simple interface evaluation is dependent on the
# input you give it. If you give it just numbers, a number is returned if the
# dimensionality of the output is 1.
print("\nSimple interface with number input (normal)")
print(simple(0.4, 0.4))
print("\nSimple interface with number input (derivative)")
print(simple(0.4, 0.4, derivative=True))