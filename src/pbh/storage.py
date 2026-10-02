"""How each cell's energy is stored: as its deviation from FRW, or whole (paper Section 7.6, the deviation form).

The integrator carries the deviation `delta E_c = E_c - Delta V_c` of each cell's energy from its FRW content (Section
7.6), which keeps a cell near the background to the last bit of its deviation. A cell far below the background loses by
it what the deviation form gains: its content is `Delta V_c + delta E_c` with `delta E_c` close to `-Delta V_c`, known
to about `eps Delta V_c`, so its density is uncertain by `eps` absolutely and by `eps / rho_c` relatively, a tenth at
`1e-15`, and nothing at all below `1e-16`, where `1 + delta_rho` rounds to zero. The void that opens near threshold
inside the infalling shell empties far below that and refills; in deviation form its rates are noise.

So each cell stores one number and a flag. Near the background the number is the deviation. Once the cell's density
falls below `TO_WHOLE = 1/4` at a step boundary it is the whole content `E_c`, and the cell goes back to its deviation
when its density rises above `TO_DEVIATION = 1/2`; the gap between the lines keeps a cell from flapping across them.
Both conversions are exact in floating point by Sterbenz's lemma (`x - y` is exact for `y / 2 <= x <= 2 y`):
`Delta V + delta E` for `rho <= 1/2`, and `E - Delta V` for `1/2 <= rho <= 2`. (A cell that rose past `2` in one
step converts with the rounding of a deviation, which is all it is then stored to.)

A cell stored whole forms from whole values everything that needs its relative precision: its density and lapse
(`derived.py`), its reconstruction slopes and theta-limiter, the HLL face states and fluxes at its two faces
(`kernels.py`), its energy rate `-(F_{c+1} - F_c) + (2 - 3 alpha) E_c`, and the pressure gradient of the velocity rows
at its faces (`equations.py`). What works in sums of deviations, the mass deviation and with it `Gammabar^2`, reads its
`E - Delta V`, which carries only the absolute rounding every deviation carries. The update is unchanged: the integrator
advances the stored number by its own rate, `delta E` by `d_xi delta E` and `E` by `d_xi E`, and the checked stepper is
the same. Each face carries one flux, `F` beside a cell stored whole and `F_FRW + delta F` elsewhere, the two agreeing
to the rounding of `F_FRW`, so the cumulative-sum bookkeeping holds to round-off.

With no cell below `1/4` nothing is stored whole and the code is the deviation form to the bit. The outermost
`KEEP_DEVIATION` cells, which the outer closure reads (Section 7.5), are always stored as deviations: the closure is
formed in deviations.
"""

import numpy as np

from pbh.layout import Layout
from pbh.state import State
from pbh.types import BoolArray, FloatArray

#: A cell is stored whole once its density falls below this fraction of the background at a step boundary.
TO_WHOLE = 0.25
#: A cell stored whole goes back to its deviation once its density rises above this.
TO_DEVIATION = 0.5
#: The outermost cells, which the outer closure reads, are never stored whole.
KEEP_DEVIATION = 2


def any_whole(whole: BoolArray | None) -> BoolArray | None:
    """The flags if any cell is stored whole, `None` if none is: the form every consumer takes them in."""
    return whole if whole is not None and bool(whole.any()) else None


def faces_beside(whole: BoolArray) -> BoolArray:
    """The faces with a cell stored whole on either side (`N + 1` of them, for `N` cells)."""
    beside = np.zeros(whole.size + 1, dtype=bool)
    beside[:-1] |= whole
    beside[1:] |= whole
    return beside


def whole_of(stored: State, whole: BoolArray | None, frw: State) -> State:
    """The state the stored numbers stand for: FRW plus the deviation, and the content itself in cells stored whole."""
    state = frw.plus(stored)
    if whole is None:
        return state
    return State(E=np.where(whole, stored.E, state.E), U=state.U, W=state.W, M_e=state.M_e)


def deviation_of(stored: State, whole: BoolArray | None, dV: FloatArray) -> State:
    """The deviation from FRW of every entry, `E - Delta V` in the cells stored whole (rounded, as any deviation is)."""
    if whole is None:
        return stored
    return State(E=np.where(whole, stored.E - dV, stored.E), U=stored.U, W=stored.W, M_e=stored.M_e)


def switched(
    dy: FloatArray, whole: BoolArray | None, rho: FloatArray, dV: FloatArray, layout: Layout
) -> tuple[FloatArray, BoolArray | None] | None:
    """Move the cells across the lines at a step boundary: the new packed vector and flags, `None` if none moves.

    Args:
        dy: The packed stored numbers.
        whole: Which cells are stored whole, `None` if none.
        rho: The cell densities of the state `dy` stands for (the derived field of its evaluation).
        dV: The shell volumes, the FRW contents.
        layout: The layout `dy` is packed by.
    """
    N, j_e = layout.N, layout.j_e
    flags = np.zeros(N, dtype=bool) if whole is None else whole
    eligible = np.zeros(N, dtype=bool)
    eligible[j_e : N - KEEP_DEVIATION] = True
    to_whole = eligible & ~flags & (rho < TO_WHOLE)  # NaN (excised) compares false
    to_deviation = flags & (rho > TO_DEVIATION)
    if not (to_whole.any() or to_deviation.any()):
        return None
    stored = layout.unpack(dy)
    E = stored.E.copy()
    E[to_whole] = dV[to_whole] + E[to_whole]  # exact for rho <= 1/2
    E[to_deviation] = E[to_deviation] - dV[to_deviation]  # exact for 1/2 <= rho <= 2
    flags = (flags | to_whole) & ~to_deviation
    return layout.pack(State(E=E, U=stored.U, W=stored.W, M_e=stored.M_e)), any_whole(flags)
