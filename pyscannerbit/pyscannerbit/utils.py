import copy

# Nested dictionary merge function
# Modified from: https://stackoverflow.com/a/7205107/1447953 
def _merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                #raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
                # Actually the behaviour we want is for a to override b, so do nothing here
                pass
        else:
            a[key] = copy.deepcopy(b[key])
    return a

