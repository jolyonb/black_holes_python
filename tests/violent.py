"""The violent states of the positivity lemma's tests (Section 7.7), shared with the engine-agreement tests.

Seeded, so that every run sees the same ones: near-vacuum cells beside dense ones, strong compression and expansion,
trapped and open excision faces, a trapped face whose chord points outward on a moving map, moving maps, grids
reaching far beyond the chord crossover, inflow at the outer face; at radiation and at the stiff fluid, both exact.
`test_positivity.py` says what each family is for.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import PRODUCTION_KERNELS
from pbh.layout import Layout
from pbh.maps import BlendMap, IdentityMap, Map, PinnedMap, SinhStretch, Zone
from pbh.outer import HeldAtFrw, OutgoingWave
from pbh.state import State
from pbh.timestep import Scheme

#: The equations of state the lemma is tested at, by label: radiation and the stiff fluid, exact.
W_CASES: dict[str, Fraction] = {"w=1/3": RADIATION, "w=1": Fraction(1)}
EOSES: dict[str, EquationOfState] = {label: EquationOfState(w) for label, w in W_CASES.items()}
RAD = "w=1/3"
EOS = EOSES[RAD]
SEEDS = range(12)
FAMILIES = ("void", "shock", "superhorizon", "stretched", "excised", "blend", "pinned", "outer_inflow")


# --- the violent states ---


@dataclass(frozen=True)
class Case:
    """One violent state: the map, layout and time it lives at, and which outer closure the lemma's face N uses.

    Attributes:
        family: The generator family.
        seed: The seed within it.
        map: The map (the blend and pinned maps move).
        layout: The layout; `j_e > 0` for the excised family.
        xi: The time.
        state: The whole state.
        closure: `"sat"` (the outgoing-wave penalty, static outer face) or `"held"` (the outer face held at FRW; the
            pinned map moves its outer face, which the SAT closure refuses).
        kind: For the excised family, `"trapped"` (static map, trapped face), `"trapped_out"` (moving map, trapped
            face whose chord points outward, `v_e > 0`) or `"open"` (static map, face not trapped); else `""`.
        eos: The equation of state (radiation unless the test says otherwise).
    """

    family: str
    seed: int
    map: Map
    layout: Layout
    xi: float
    state: State
    closure: str
    kind: str = field(default="")
    eos: EquationOfState = field(default=EOS)


def _geometry(m: Map, xi: float, N: int) -> Geometry:
    return Geometry.of(*m.radii(xi, N))


def violent_case(family: str, seed: int, w: str = RAD) -> Case:
    """A seeded violent state of the family at the equation of state `EOSES[w]`, built so that the stage accepts it
    (`rho > 0`, `Gammabar^2 > 0`). The seeds are the same at every `w`; only the physics that depends on it differs.

    The excised family has three kinds by seed: 0-5 `"trapped"` (static map, `U_je < -Gammabar_je`), 6-8
    `"trapped_out"` (the blend map pinned at the face, `d_xi X ~ -alpha X`, full tension `q / rho = -w` in the first
    retained cell and `e^phi |U| < w X`, so that the face is trapped yet its chord `v_e` is positive: Section 8.3
    shows this needs a moving map) and 9-11 `"open"` (static map, `Theta + a > 0` at the face, so that even the
    acoustic bounds give `Lambda+ > 0` there and the rule `rho^L := rho^R` is what keeps the flux the cell's own).
    """
    eos = EOSES[w]
    alpha, lapse = float(eos.alpha), eos.lapse_exponent
    rng = np.random.default_rng([FAMILIES.index(family), seed])
    N, j_e, closure, kind = 40, 0, "sat", ""
    xi = 1.0
    m: Map = IdentityMap(8.0)
    if family == "superhorizon":  # the chord crossover at xi = 0 is 1.7: most of the grid is beyond it
        m, xi = IdentityMap(24.0), 0.0
    elif family == "stretched":
        m = SinhStretch(12.0, 3.0)
    elif family == "blend":  # inside the ramp: the interior moves, the outer face does not
        m = BlendMap(IdentityMap(8.0), alpha, (Zone(xi_on=1.0, tau_on=0.3, x_t=0.45, Delta_t=0.3),))
        xi = 1.0 + float(rng.uniform(0.05, 0.6))
    elif family == "pinned":  # every face moves, the outer one too
        m, closure = PinnedMap(IdentityMap(8.0), alpha, xi_on=1.0), "held"
        xi = 1.0 + float(rng.uniform(0.05, 0.6))
    elif family == "excised":
        kind = "trapped" if seed < 6 else ("trapped_out" if seed < 9 else "open")
        if kind == "trapped_out":  # past the ramp, pinned well beyond the face
            m = BlendMap(IdentityMap(8.0), alpha, (Zone(xi_on=1.0, tau_on=0.3, x_t=0.7, Delta_t=0.2),))
            xi = 2.0
            j_e = int(rng.integers(12, 19))
        else:
            j_e = int(rng.integers(3, 11))
    geo = _geometry(m, xi, N)
    X = geo.X
    bg2 = Background.at(eos, xi).Gammabar2

    # densities: dense cells between 0.05 and 20, a quarter of them near vacuum, 1e-9 to 1e-4, and in every third seed
    # down to 5e-13, where a content stored as its deviation is resolved only to eps / rho (Section 7.2)
    rho = np.exp(rng.uniform(math.log(0.05), math.log(20.0), N))
    vacuum = rng.random(N) < 0.25
    deepest = -12.3 if seed % 3 == 2 else -9.0
    rho[vacuum] = 10.0 ** rng.uniform(deepest, -4.0, int(np.sum(vacuum)))
    # velocities: U = X (1 + delta) with a random peculiar part of order one, or a profile
    delta = rng.uniform(-2.5, 2.5, N + 1)
    if family == "void":  # pull apart about a random radius: inward inside, outward outside
        X0 = float(rng.uniform(1.5, 5.0))
        delta = 3.0 * np.tanh((X - X0) / 0.3) + rng.uniform(-0.5, 0.5, N + 1)
    elif family == "shock":  # converge on a random radius: outward inside, inward outside
        X0 = float(rng.uniform(1.5, 5.0))
        delta = -3.0 * np.tanh((X - X0) / 0.3) + rng.uniform(-0.5, 0.5, N + 1)
    elif family == "superhorizon":
        delta = rng.uniform(-0.8, 0.8, N + 1)
    elif family == "pinned":
        # The outer cells at FRW. The pinned map moves the outer face, which only HeldAtFrw allows, and both are
        # outside the lemma's scope, which covers the SAT closure on a static outer face (Section 7.7). Holding the last
        # cells at FRW keeps face N at its FRW value, so that the family tests the moving interior, not the closure.
        rho[N - 4 :] = 1.0
        delta[N - 3 :] = 0.0
    elif family == "outer_inflow":  # a near-empty last cell beside a dense one, the flow at face N inward
        rho[N - 1] = 10.0 ** rng.uniform(-9.0, -5.0)
        rho[N - 2] = float(rng.uniform(2.0, 20.0))
        delta[N] = -float(rng.uniform(0.5, 3.0))
    U = X * (1.0 + delta)

    E = np.full(N, np.nan)
    M_e = 0.0
    guard_from = max(j_e, 1)
    if kind == "trapped":  # a trapped excision face: M / X = 2 e^(2 (1 - alpha) xi) there, infall faster than sound
        M_e = 2.0 * bg2 * float(X[j_e])
        U[j_e] = -1.5 * math.sqrt(bg2)
    elif kind == "trapped_out":  # trapped, Gammabar = |U| / 1.05, and full tension in the first retained cell
        rho[j_e] = 20.0
        U[j_e] = -0.6 * float(X[j_e])
        M_e = float(X[j_e]) * (bg2 + U[j_e] ** 2 * (1.0 - 1.0 / 1.05**2))
        U[j_e + 1] = 3.0 * float(X[j_e + 1])
        guard_from = j_e + 1
    elif kind == "open":  # e^phi U = 2 X at the face: Theta = alpha X > 0, not trapped
        U[j_e] = 2.0 * float(X[j_e]) * float(rho[j_e]) ** (-lapse)
        M_e = float(X[j_e]) ** 3 * float(rng.uniform(0.5, 1.5))
    if j_e > 0:
        U[:j_e] = np.nan
    else:
        U[0] = 0.0
    E[j_e:] = rho[j_e:] * geo.dV[j_e:]
    # Gammabar^2 = bg2 + U^2 - M / X >= bg2 / 5 at every retained face: raise |U| where the mass needs it
    M = np.full(N + 1, np.nan)
    M[j_e] = M_e
    M[j_e + 1 :] = M_e + 3.0 * np.cumsum(E[j_e:])
    for j in range(guard_from, N + 1):
        need = float(M[j] / X[j]) - 0.8 * bg2
        if U[j] ** 2 < need:
            U[j] = math.copysign(math.sqrt(need) * 1.05, U[j] if U[j] != 0.0 else 1.0)
    W = float(rng.normal(0.0, 0.1)) if closure == "sat" else 0.0
    return Case(family, seed, m, Layout(N, j_e), xi, State(E=E, U=U, W=W, M_e=M_e), closure, kind, eos)


def production_scheme(case: Case) -> Scheme:
    """The production scheme on the case: the production kernels and closure (the held face on the pinned map)."""
    outer = OutgoingWave() if case.closure == "sat" else HeldAtFrw()
    return Scheme(case.eos, case.map, case.layout, outer, PRODUCTION_KERNELS)
