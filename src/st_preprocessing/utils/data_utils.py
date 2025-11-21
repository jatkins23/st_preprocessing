import numpy as np

def flatten_array(x):
    """
    Flattens a numpy array that may contain nested lists or arrays.
    Works recursively and returns a 1D numpy array.
    """
    # Convert to a Python list first (handles both np.ndarray and list)
    if isinstance(x, np.ndarray):
        x = x.tolist()
    
    # Recursively flatten any nested lists
    def _flatten(lst):
        for i in lst:
            if isinstance(i, (list, np.ndarray)):
                yield from _flatten(i)
            else:
                yield i

    return np.array(list(_flatten(x)), dtype=object)
