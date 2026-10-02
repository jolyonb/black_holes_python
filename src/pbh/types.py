"""The array types the code uses: a numpy array of doubles, and the NaN-filled array every field starts as.

The paper's Section 7.2 presumes double precision throughout (in single precision the round-off floors it quotes would
stand above the truncation error of the scheme), so no other floating-point dtype appears anywhere. The one other array
is a mask of booleans, such as the cells stored whole (`storage.py`).
"""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]


def nan_array(n: int) -> FloatArray:
    """An array of `n` NaNs: a field before its retained entries are filled in (the NaN convention of `layout.py`).

    Formed as `np.empty` and a fill rather than by `np.full`, whose Python wrapper costs twice as much; a stage makes
    about forty of these.
    """
    a = np.empty(n)
    a.fill(np.nan)
    return a


def read_only(*arrays: FloatArray) -> None:
    """Mark arrays read-only that several consumers share (a frame's geometry, a map's kept values).

    A stray in-place write then raises instead of silently changing what the next reader sees.
    """
    for a in arrays:
        a.flags.writeable = False
