import numpy as np
try:
    import cupy
    if cupy.cuda.is_available():
        np = cupy
except:
    pass