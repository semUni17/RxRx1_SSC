import numpy as np


class ToNPArray:
    def __call__(self, x):
        x = np.array(x, dtype=np.uint8)
        return x
