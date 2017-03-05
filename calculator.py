# -----------------------------------------------------------------------------
# calculator.py
# -----------------------------------------------------------------------------
#
# Initially: There are 1000014 function calls in 3.204 seconds measured by CPython
# And Line_profiler indicates that most of the runtime were spent on the for
# loops.
# After using np.add(), np.multiply() and np.sqrt() from numpy package, by cPython, there are 10284 function calls in 0.125 seconds
#Speedup: 3.204 / 0.125 = 25.632x

import numpy as np



def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = np.multiply(x,x)
    yy = np.multiply(y, y)
    zz = np.add(xx, yy)
    return np.sqrt(zz)
