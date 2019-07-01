"""
Show testfunctions available in the functions module, grouped per function
group. Functions can be part of multiple groups.
"""
from high_dimensional_sampling import functions

feeder = functions.FunctionFeeder()

groups = [
    'optimisation', 'posterior', 'with_derivative', 'no_derivative', 'bounded',
    'unbounded'
]

for group in groups:
    print(group.upper())
    print('-'*32)
    feeder.load_function_group(group)
    for f in feeder:
        print(type(f).__name__)
    feeder.reset()
    print('='*32)
