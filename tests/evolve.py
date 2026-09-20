"""One evolution loop for the tests: the deviation from FRW advanced by RK4 at the Courant step (Section 7.6)."""

from pbh.state import State
from pbh.timestep import COURANT_NUMBER, Integrator, Scheme, advance, courant_step


def evolve(sch: Scheme, state: State, xi: float, xi_end: float) -> State:
    """Advance `state` from `xi` to `xi_end` on a static map, landing exactly on `xi_end`."""
    geo = sch.frame(xi).geo
    dy = sch.layout.pack(state) - sch.frw(xi)
    while xi < xi_end - 1e-12:
        res = sch.evaluate(xi, sch.frw(xi) + dy)
        dxi = min(courant_step(res, geo, sch.layout, COURANT_NUMBER), xi_end - xi)
        dy = advance(sch, Integrator.RK4, xi, dy, dxi)
        xi += dxi
    return sch.layout.unpack(sch.frw(xi_end) + dy)
