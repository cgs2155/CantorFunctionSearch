import numpy as np

def cantors(n):
    if n == 0:
        def f(x):
            return x
    else:
        prev = cantors(n-1)
        def f(x):
            # For scalars: if 3*x is in (1,2), return 0.5; otherwise compute recursively.
            if 1 < 3*x < 2:
                return 0.5
            elif 3*x < 1:
                # x//0.5 is equivalent to floor(x/0.5)
                return prev(3*x) / 2
            else:
                return 0.5+ prev(3*x-2) / 2

    return np.vectorize(f)
    