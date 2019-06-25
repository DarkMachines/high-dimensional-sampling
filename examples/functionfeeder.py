from high_dimensional_sampling import functions

feeder = functions.FunctionFeeder()
feeder.load_function("Sphere")
feeder.load_function("Himmelblau")
feeder.load_function("Booth")

for f in feeder:
    print(type(f).__name__)