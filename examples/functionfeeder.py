from high_dimensional_sampling import functions

feeder = functions.FunctionFeeder()

groups = [
    'optimisation', 'posterior', 'with_derivative', 'no_derivative', 'bounded',
    'unbounded'
]

for group in groups:
    print(group.upper())
    print('-'*32)
    feeder.load_functions(group)
    for f in feeder:
        print(type(f).__name__)
    feeder.reset()
    print('='*32)

