"""One evolution loop for the tests: the deviation from FRW advanced by RK4 at the Courant step (Section 7.6)."""

from pbh.state import State
from pbh.timestep import COURANT_NUMBER, Scheme, advance, courant_step


def evolve(sch: Scheme, state: State, xi: float, xi_end: float) -> State:
    """Advance `state` from `xi` to `xi_end`, landing exactly on `xi_end`.

    On a moving map the Courant step is taken from the geometry at the start of each step, since pinned cells shrink.
    """
    dy = sch.layout.pack(state) - sch.frw(xi)
    while xi < xi_end - 1e-12:
        geo = sch.frame(xi).geo
        res = sch.evaluate_deviation(xi, dy)
        dxi = min(courant_step(res, geo, sch.layout, COURANT_NUMBER), xi_end - xi)
        dy = advance(sch, xi, dy, dxi)
        xi += dxi
    return sch.layout.unpack(sch.frw(xi_end) + dy)
