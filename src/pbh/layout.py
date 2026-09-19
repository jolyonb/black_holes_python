"""Which entries are unknowns, and how they are packed into one vector (paper Table tab:num:layout; Section 8.3).

The fields live on the grid of `geometry.py`, with the same picture and the same indexing:

    face j       0        1        2        3        4 = N
    cell c           0        1        2        3
    E_c              *        *        *        *          cell energies E_c, N of them
    U_j        (=0)       *        *        *        *     face velocities U_j, evolved at faces 1..N
    W                                                       one scalar, the incoming amplitude at face N

Face `0` is the origin and carries no unknown: `U_0 = 0` identically (Section 7.2, "at the origin nothing is
imposed"). Every array in the code is nevertheless full length, `E` with `N` entries indexed by `c` and `U` with
`N + 1` entries indexed by `j`, so that `E[c]` is `E_c` and `U[j]` is `U_j` exactly as printed.

After a horizon forms the cells inside the excision face `j_e` are dropped (Section 8.3): the retained unknowns are
`E_c` for `c >= j_e`, `U_j` for `j >= j_e`, the outer scalar `W`, and one new scalar `M_e`, the mass inside the
excision face, `M_{j_e}`. The arrays stay full length; the entries below `j_e` are NaN and are never read, so that any
stencil that reaches into the excised region announces itself through the finiteness assertion of the driver rather
than silently using a stale value.

The integrator works on one flat vector, in the order

    [ E_c for retained cells | U_j for evolved faces | M_e (only if excised) | W ]

and `Layout` is the only place that knows this order or which entries are retained; everything else asks it.
"""

from dataclasses import dataclass

import numpy as np

from pbh.state import State
from pbh.types import FloatArray


@dataclass(frozen=True)
class Layout:
    """The retained ranges of the grid and the packing of the unknowns into one vector.

    Attributes:
        N: The number of cells.
        j_e: The index of the excision face; `0` before excision, when the origin face is the inner edge and
            carries no unknown.
    """

    N: int
    j_e: int = 0

    def __post_init__(self) -> None:
        """The excision face must lie inside the grid: `0 <= j_e < N` (Section 8.2 needs `1 <= j_e < j_*`)."""
        if self.N < 2:
            raise ValueError(f"need at least two cells, got N = {self.N}")
        if not 0 <= self.j_e < self.N:
            raise ValueError(f"the excision face must satisfy 0 <= j_e < N, got j_e = {self.j_e} with N = {self.N}")

    @property
    def excised(self) -> bool:
        """Whether cells have been dropped, in which case `M_e` is an unknown."""
        return self.j_e > 0

    @property
    def cells(self) -> slice:
        """The retained cells, `c = j_e .. N-1`."""
        return slice(self.j_e, self.N)

    @property
    def faces(self) -> slice:
        """The retained faces, `j = j_e .. N`, the innermost being the origin or the excision face."""
        return slice(self.j_e, self.N + 1)

    @property
    def faces_evolved(self) -> slice:
        """The faces whose velocity is an unknown: `j = max(j_e, 1) .. N`, since `U_0 = 0` is never evolved."""
        return slice(max(self.j_e, 1), self.N + 1)

    @property
    def size(self) -> int:
        """The length of the packed vector."""
        n_cells = self.N - self.j_e
        n_faces = self.N + 1 - max(self.j_e, 1)
        return n_cells + n_faces + int(self.excised) + 1

    def pack(self, state: State) -> FloatArray:
        """The state as one flat vector, `[E (retained) | U (evolved) | M_e if excised | W]`."""
        scalars = [state.M_e, state.W] if self.excised else [state.W]
        return np.concatenate((state.E[self.cells], state.U[self.faces_evolved], scalars))

    def unpack(self, y: FloatArray) -> State:
        """The state from a flat vector packed by `pack`, full-length arrays with NaN below `j_e` and `U_0 = 0`."""
        if y.shape != (self.size,):
            raise ValueError(f"expected a packed vector of length {self.size}, got shape {y.shape}")
        n_cells = self.N - self.j_e
        n_faces = self.N + 1 - max(self.j_e, 1)
        E = np.full(self.N, np.nan)
        E[self.cells] = y[:n_cells]
        U = np.full(self.N + 1, np.nan)
        U[self.faces_evolved] = y[n_cells : n_cells + n_faces]
        if not self.excised:
            U[0] = 0.0
        M_e = y[n_cells + n_faces] if self.excised else 0.0
        return State(E=E, U=U, W=float(y[-1]), M_e=float(M_e))
