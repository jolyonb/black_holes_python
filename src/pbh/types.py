"""The one array type the code uses: a numpy array of doubles.

The paper's Section 7.2 presumes double precision throughout (in single precision the round-off floors it quotes would
stand above the truncation error of the scheme), so no other dtype appears anywhere.
"""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
