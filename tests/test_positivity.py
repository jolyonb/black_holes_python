"""Property tests of the positivity lemma for the energy row (analysis/v4/positivity/lemma.md), written test-first.

The lemma makes the energy row of eq:num:energy positivity-preserving in two layers. Layer 1 (semi-discrete): with
HLL bounds that bracket the one-sided chord speeds `v = F(rho) / (X^2 rho)`, every face flux is
`X^2 [A rho^L - B rho^R]` with `A, B >= 0`, so a cell's rate is non-negative gains from its neighbours plus
`sigma E_c` minus a loss `K_c E_c` through its own two face values: no cell is drained by a neighbour's content.
Layer 2 (fully discrete): a forward-Euler step with `dxi (K_c - sigma) < 1` in every cell keeps every content
positive, `E'_c >= (1 - dxi (K_c - sigma)) E_c`.

Everything the lemma needs is recomputed here from what the scheme itself reports (its reconstructed face densities,
its `Theta` and `a`, its viscous pressure `q`, its flux and its rate), with the lemma's formulas and nothing from the
prototype:

    v(rho)      = alpha ((1 + w) rho^(-w/(1+w)) U - X) - d_xi X + alpha (rho^(-w/(1+w)) U - X) (q / rho)_cell    (V)
    Lambda+     = max(Theta + a, v^L, v^R, 0),   Lambda- = min(Theta - a, v^L, v^R, 0)                        (B)
    A           = Lambda+ (v^L - Lambda-) / (Lambda+ - Lambda-),   B = (-Lambda-) (Lambda+ - v^R) / (Lambda+ - Lambda-)
    L_c         = X_(c+1)^2 A_(c+1) rho^+_c + X_c^2 B_c rho^-_c,   K_c = L_c / E_c                            (L1)
    lam^-_c     = (X_(c+1)^2 - sbar_c) / (X_(c+1)^2 - X_c^2),   rho_c = lam^- rho^-_c + lam^+ rho^+_c           (ZS)

with the lemma's face rules: `F_0 = 0` at the origin; at an excision face `F = X^2 v rho^-` of the first retained
cell's own face value; at the static outer face `F_N = X_N^2 v*_N rho_hat_N` with `0 < rho_hat_N <= rho_(N-1)/lam^+`
carried at its own lapse, `v*_N` the chord at the SAT velocity `U*_N = U_N - (tau_rho X_N / 2) pen` (at `U_N` itself
on the held face of the pinned map). The positivity checks read face N from the scheme's own `F_N` (a loss
`max(F_N, 0)` bounded by what any admissible face value could carry, `check_face_n`), so they hold for the donor as
for the reconstruction; only `check_flux_face_n` names the reconstructed value; `check_flux_interior` pins the
interior faces' flux specification. `check_pair_n` pins the rest of face N (lemma.md Sec. 5.4): face N has ONE state,
the pair `(rho_hat_N, e^phi(rho_hat_N))` of that same reconstructed value, which the stage must report as its derived
`<rho>_N`, `<e^phi>_N`, form its face-N speeds from, and feed to the velocity row at N; it reads only production
outputs (the derived fields, the speeds, the deviation rate), and forms what they must equal with production's own
`pbh.equations.speeds` and `pbh.outer.OutgoingWave.rows`. `check_pair_n_kernels_off` asserts the same of the centred
base scheme (the kernels off, the test switch), which reports no reconstruction: it forms `rho_hat_N` itself, from
the theta-limited one-sided `s`-slope of cells `N-2` and `N-1` at the scheme's own theta.

The states are violent on purpose (near-vacuum cells beside dense ones, strong compression and expansion, trapped
and open excision faces, a trapped face whose chord points outward on a moving map, moving maps, grids reaching far
beyond the chord crossover, inflow at the outer face) and seeded, so that every run sees the same ones. The chord
crossover is `X_c = e^((1-alpha) xi) / sqrt(w)`, where the FRW pressure-work chord `alpha w X` equals the sound speed
`a`: `3 tau`, three sound horizons `tau = e^(xi/2)/sqrt3` of eq:lin:tau for radiation, and the acoustic bounds miss the
chord everywhere beyond it. Today's scheme fails the lemma on some of them, and those tests are marked
`xfail(strict=True)` naming the ingredient it lacks: when the scheme changes they XPASS, the suite goes red, and the
marker must come off. That is not enough on its own: a marked case that still fails stays green, so a bite with a
defect would hide behind the markers it did not flip. After a bite, EVERY marker of the checks the bite implements is
deleted wholesale (`IMPLEMENTS` names them), not only those that XPASS; any case of those checks that still fails is a
defect of the bite, never a marker to keep. `test_no_marker_outlives_the_ingredient_that_implements_its_check` enforces
it, detecting the ingredients from the scheme's behaviour. `analysis/v4/positivity/proptest_prototype.py` runs the same
generator and the same checks with the prototype of the lemma installed and shows they pass there, that a variant
without the chord bounds fails, and, in its post-bite mode, that six defective bites fail with the markers removed.
Not only the strict xfails change with the scheme: the anchor `test_the_coefficient_identity_reproduces_todays_flux_
with_its_own_bounds` audits the flux with today's bounds `Theta +- a` and fails on every family once the bounds are
the chord bounds (proptest_prototype.py reports it). The scheme-change bite must replace it: its role, the anchor that
the audit's algebra is transcribed right, passes to `test_every_interior_flux_follows_the_lemmas_flux_specification`,
which then holds on every family.

The states are generated at two equations of state, radiation `w = 1/3` and the stiff fluid `w = 1`, both exact
Fractions, because the lemma's statements are general in `w` (the source `sigma = 3 alpha w`, the lapse exponent
`-w/(1+w)` of the chords and of the theta bound). The near-vacuum cells reach down to `5e-13`, the owner's abort
threshold (lemma.md Sec. 8), in a third of the seeds, so the tolerances are calibrated where round-off is largest.

Layer 2's *checked stepper* (an explicit Runge-Kutta method, RK4 recommended and SSPRK(3,3) the documented
alternative, with a check of every stage input, every stage evaluation and the result; halving on a positivity or
non-finite failure; the named aborts, `Gammabar^2 <= 0` at a stage or at the result among them) is not tested here,
because its production API does not exist yet. Its acceptance list is lemma.md Sec. 4.5, and its contract is written
as `stepper_contract` in `analysis/v4/positivity/proptest_prototype.py`, parametrized by the stepper, its Butcher
tableau and the evaluation it calls, and run there against the prototype for both methods. It must be ported into
`code/tests` with the production bite, pointed at the production stepper.
"""

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pytest

from pbh.derived import Derived, NotHyperbolicError
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.equations import DerivsResult, Speeds, speeds
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS
from pbh.layout import Layout
from pbh.maps import BlendMap, IdentityMap, Map, PinnedMap, SinhStretch, Zone
from pbh.outer import PRODUCTION_STRENGTHS, HeldAtFrw, OuterInputs, OutgoingWave, characteristic_pair
from pbh.state import State, deviation_from_frw
from pbh.timestep import Scheme, courant_step
from pbh.types import FloatArray

#: The equations of state the lemma is tested at, by label: radiation and the stiff fluid, exact.
W_CASES: dict[str, Fraction] = {"w=1/3": RADIATION, "w=1": Fraction(1)}
EOSES: dict[str, EquationOfState] = {label: EquationOfState(w) for label, w in W_CASES.items()}
RAD = "w=1/3"
EOS = EOSES[RAD]

#: The least face value, as a fraction of its cell's density, that the lemma's theta-limiter guarantees (the owner's
#: range is 0.1 to 0.3): it bounds a face lapse by `THETA_MIN^(-w/(1+w))` times its cell's.
THETA_MIN = 0.1
#: The certificate step is this fraction of `1 / max(K - sigma)`, so the lemma promises `E' >= 0.1 E`.
CERTIFICATE_FRACTION = 0.9
#: The round-off tolerance of the Zhang-Shu mean, in units of `eps (1 + rho_c)`: the reconstruction works in
#: `rho - 1`, so a face value carries an absolute error of a few `eps`. Measured worst with the prototype (both w, both
#: theta; proptest_prototype.py prints it): about 1.7, so 8 leaves a margin of about 5. In absolute terms the tolerance
#: is about 2e-15 for a near-empty cell, so a 10 per cent error in the mean of a cell at `rho_c < 1e-12` (1e-13 or
#: more) is caught; the round 2 tolerance, `1e-13 (1 + rho_c)`, was blind to it. The face floor of 1e-12 breaks the
#: mean by far more than this in a near-empty cell.
ZS_ROUNDOFF_EPS = 8.0
#: The round-off tolerance of the theta bound, in units of `eps (1 + rho_c)`: the reconstruction forms face values as
#: `1 + (delta rho_c + t offset)`, so a face value that the limiter puts exactly at `theta rho_c` carries an absolute
#: error of a few `eps`, which is `eps / rho_c` relative. Measured worst with the prototype at `theta = THETA_MIN`:
#: 0.52 `eps (1 + rho_c)`, so 8 leaves a margin of 15.
THETA_ROUNDOFF_EPS = 8.0
SEEDS = range(12)
#: Round-off tolerances of the gain and flux identities, in units of `eps` times the size of the terms (`cell_scale`,
#: `face_scale`). Measured worst over every family, seed and `w` with the prototype installed
#: (analysis/v4/positivity/proptest_prototype.py prints them; lemma.md claim 36): 0.25 eps for the gain and 19.7 eps for
#: the flux (an interior face of the moving blend map), so 16 and 128 leave margins of 64 and 6.5.
GAIN_ROUNDOFF_EPS = 16.0
FLUX_ROUNDOFF_EPS = 128.0
EPS = float(np.finfo(float).eps)
#: The content-relative drain bound: whatever a neighbour drains from a cell in one Courant step (C = 1) must be below
#: this fraction of its content, up to the round-off of the content itself and of the step's rate,
#: `CONTENT_ROUNDOFF_EPS eps (dV_c + dxi cell_scale)`: near vacuum the stored content is known only to `eps dV_c`
#: (lemma.md Sec. 8), and a gain that is zero to round-off of the dense neighbours' terms is `eps cell_scale`, which
#: over the step can exceed `1e-6` of a content at `5e-13` (measured with the prototype: 9.6e-6 of the content, which is
#: 0.13 `eps (dV_c + dxi cell_scale)`, so 8 leaves a margin of 60; the pure content bound failed there). A drain of
#: `1e-13 X^2 (Lambda+ - Lambda-) rho_dense` injected at every face is caught in every family but the shock, whose
#: empty cells the converging flow feeds faster than the drain empties them.
DRAIN_PER_STEP = 1e-6
CONTENT_ROUNDOFF_EPS = 8.0
#: The round-off tolerance of the face-N pair (`check_pair_n`), in units of `eps`: the face value is formed as
#: `1 + (delta rho + offset)`, so it is known to an absolute `eps (1 + rho_hat)`, and whatever is formed from it
#: (its lapse, the face-N speeds, the velocity row) is compared within the change that this many such `eps` in the
#: density make, plus this many `eps` of the quantity's own size.
PAIR_ROUNDOFF_EPS = 16.0
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
    retained cell and `e^phi |U| < w X`, so that the face is trapped yet its chord `v_e` is positive: lemma.md Sec. 5.2
    shows this needs a moving map) and 9-11 `"open"` (static map, `Theta + a > 0` at the face, so that even today's
    bounds give `Lambda+ > 0` there and the rule `rho^L := rho^R` is what keeps the flux the cell's own).
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
    # down to 5e-13, the owner's abort threshold, where the content is resolved only to eps / rho (lemma.md Sec. 8)
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
        # The outer cells at FRW. HeldAtFrw's F_N still carries the extrapolated <rho>_N at the extrapolated lapse, and
        # the pinned map moves the outer face; both are outside the lemma's scope (lemma.md Sec. 1.5: the certificate
        # covers the SAT closure on a static outer face only). Holding the last cells at FRW keeps that extrapolation
        # at its FRW value so that the family tests the moving interior, not the out-of-scope closure.
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
    """Today's scheme on the case: the production kernels and the production closure (held on the pinned map)."""
    outer = OutgoingWave() if case.closure == "sat" else HeldAtFrw()
    return Scheme(case.eos, case.map, case.layout, outer, PRODUCTION_KERNELS)


# --- the lemma, recomputed from what the scheme reports ---


@dataclass(frozen=True)
class Audit:
    """The lemma's quantities for one evaluation of one scheme on one case (face arrays `N + 1`, cell arrays `N`).

    Attributes:
        F: The scheme's own energy flux.
        F_lemma: The flux specification: `X^2 [A rho^L - B rho^R]` with the chord bounds (interior and excision faces),
            and at face N `X_N^2 v*_N rho_hat_N` with `rho_hat_N` the reconstructed value `rho^L_N`.
        face_scale: The size of the terms at each face, for round-off tolerances.
        rho_minus: Each retained cell's reconstructed value at its inner face.
        rho_plus: ... and at its outer face.
        rho: The cell densities.
        lam_minus: The Zhang-Shu weight of the inner face value.
        E: The contents.
        dV: The cell volumes: the content of a cell is known only to about `eps dV` (lemma.md Sec. 8).
        rate: The scheme's whole rate of the contents.
        loss: The lemma's loss through the cell's own face values, `L_c`; at face N the scheme's own `max(F_N, 0)`.
        loss_bound: `L_c` bounded through (ZS), `[X_(c+1)^2 A_(c+1) / lam^+ + X_c^2 B_c / lam^-] rho_c`, with face N's
            term `faceN_bound`.
        cell_scale: The size of the terms of each cell's rate.
        faceN_bound: The most any admissible face value `0 < r <= rho_(N-1) / lam^+_(N-1)` at its own lapse can carry
            out through face N, `X_N^2 max(0, sup_r r v*_N(r))`: the own-content bound on the last cell's loss there.
        dxi: The Courant step at `C = 1`, longer than any step the scheme takes.
        defects: Face values or chords the audit could not form (non-positive densities, non-finite speeds).
        A: The lemma's coefficient of `X^2 rho^L` at each face.
        B: ... of `-X^2 rho^R`.
    """

    F: FloatArray
    F_lemma: FloatArray
    face_scale: FloatArray
    rho_minus: FloatArray
    rho_plus: FloatArray
    rho: FloatArray
    lam_minus: FloatArray
    E: FloatArray
    dV: FloatArray
    rate: FloatArray
    loss: FloatArray
    loss_bound: FloatArray
    cell_scale: FloatArray
    faceN_bound: float
    dxi: float
    defects: list[str]
    A: FloatArray
    B: FloatArray


def chord(
    eos: EquationOfState, rho: FloatArray, U: FloatArray, X: FloatArray, X_xi: FloatArray, q_over_rho: FloatArray
) -> FloatArray:
    """The one-sided chord speed (V): the flux at this side's density over `X^2 rho`, at that density's lapse.

    NaN where the side's density is not positive (its lapse is undefined); the audit reports those as defects.
    """
    alpha, w = float(eos.alpha), float(eos.w)
    positive = rho > 0.0
    ephi = np.where(positive, np.where(positive, rho, 1.0) ** eos.lapse_exponent, np.nan)
    return alpha * ((1.0 + w) * ephi * U - X) - X_xi + alpha * (ephi * U - X) * q_over_rho


def most_out_of_face_n(eos: EquationOfState, U_star: float, X_N: float, X_xi_N: float, r_max: float) -> float:
    """`max(0, sup_{0 < r <= r_max} r v*(r))`, `r v*(r) = alpha (1 + w) U* r^(1/(1+w)) - (alpha X_N + X_xi_N) r`.

    Concave in `r` when `b = alpha X_N + X_xi_N > 0`, with its maximum at `r* = (U* / X_N')^((1+w)/w)`,
    `X_N' = b / alpha`; increasing when `b <= 0 < U*`; never positive when `U* <= 0 <= b`.
    """
    alpha, w = float(eos.alpha), float(eos.w)
    b = alpha * X_N + X_xi_N

    def g(r: float) -> float:
        return alpha * (1.0 + w) * U_star * r ** (1.0 / (1.0 + w)) - b * r

    candidates = [0.0, g(r_max)]
    if U_star > 0.0 and b > 0.0:
        r_star = (alpha * U_star / b) ** ((1.0 + w) / w)
        if r_star < r_max:
            candidates.append(g(r_star))
    return max(candidates)


def audit(sch: Scheme, case: Case, bounds: str = "chord") -> Audit:
    """Evaluate the scheme on the case and recompute the lemma's quantities from its outputs.

    `bounds="chord"` is the lemma's (B); `bounds="acoustic"` takes the scheme's own `Theta +- a` instead, with which
    (C) must reproduce today's flux at the interior and excision faces exactly: the anchor that the algebra here is
    transcribed right.
    """
    lay = sch.layout
    N, j_e = lay.N, lay.j_e
    eos = case.eos
    alpha, w, sigma = float(eos.alpha), float(eos.w), eos.energy_source_rate
    y = lay.pack(case.state)
    res = sch.evaluate(case.xi, y)
    k = res.kernels
    if k is None:  # not an assert: a strict xfail must be satisfied only by the lemma's own assertions
        raise RuntimeError("the kernels are off; the lemma's audit needs the reconstruction and the viscous pressure")
    geo = sch.frame(case.xi).geo
    X, X_xi, X2, sbar = geo.X, geo.X_xi, geo.X**2, geo.sbar
    U = case.state.U
    rho = res.derived.rho
    q_over_rho = k.q / rho
    defects: list[str] = []

    # one-sided chords at the retained faces below N: interior faces see cells c-1 and c, the excision face its first
    # retained cell on both sides
    vL, vR = np.full(N + 1, np.nan), np.full(N + 1, np.nan)
    inner = np.arange(j_e + 1, N)
    vL[inner] = chord(eos, k.rho_L[inner], U[inner], X[inner], X_xi[inner], q_over_rho[inner - 1])
    vR[inner] = chord(eos, k.rho_R[inner], U[inner], X[inner], X_xi[inner], q_over_rho[inner])
    if j_e > 0:
        e = np.array([j_e])
        vL[j_e] = vR[j_e] = chord(eos, k.rho_R[e], U[e], X[e], X_xi[e], q_over_rho[e])[0]
    faces = np.arange(max(j_e, 1), N)
    for j in faces[~(np.isfinite(vL[faces]) & np.isfinite(vR[faces]))]:
        defects.append(f"face {j}: a face density is not positive ({k.rho_L[j]:.3g}, {k.rho_R[j]:.3g})")
    Theta, a = res.speeds.Theta, res.speeds.a
    Lp, Lm = np.full(N + 1, np.nan), np.full(N + 1, np.nan)
    if bounds == "chord":
        Lp[faces] = np.fmax.reduce([Theta[faces] + a[faces], vL[faces], vR[faces], np.zeros(faces.size)])
        Lm[faces] = np.fmin.reduce([Theta[faces] - a[faces], vL[faces], vR[faces], np.zeros(faces.size)])
    else:
        Lp[faces] = np.maximum(Theta[faces] + a[faces], 0.0)
        Lm[faces] = np.minimum(Theta[faces] - a[faces], 0.0)
    A, B = np.full(N + 1, np.nan), np.full(N + 1, np.nan)
    A[faces] = Lp[faces] * (vL[faces] - Lm[faces]) / (Lp[faces] - Lm[faces])
    B[faces] = -Lm[faces] * (Lp[faces] - vR[faces]) / (Lp[faces] - Lm[faces])
    rho_L, rho_R = k.rho_L, k.rho_R
    F_lemma = np.full(N + 1, np.nan)
    F_lemma[faces] = X2[faces] * (A[faces] * rho_L[faces] - B[faces] * rho_R[faces])
    if j_e == 0:
        F_lemma[0] = 0.0

    # face N: the SAT velocity, the specification's face value (flux check only), and the own-content bound
    U_N, X_N, X_xi_N = float(U[N]), float(X[N]), float(X_xi[N])
    if case.closure == "sat":
        c_s = Background.at(eos, case.xi).c_s
        _, u_minus = characteristic_pair(float(res.derived.delta_U[N]), float(res.derived.delta_rho[N - 1]), X_N, c_s)
        U_star = U_N - 0.5 * PRODUCTION_STRENGTHS.tau_rho * X_N * (u_minus - case.state.W)
    else:
        U_star = U_N
    rho_hat = float(rho_L[N])  # the specification: the last cell's reconstructed value at face N
    if rho_hat > 0.0:
        v_star = alpha * ((1.0 + w) * rho_hat**eos.lapse_exponent * U_star - X_N) - X_xi_N
    else:
        defects.append(f"face {N}: the reconstructed face value {rho_hat:.3g} is not positive")
        v_star = math.nan
    F_lemma[N] = X_N**2 * v_star * rho_hat
    A[N], B[N] = v_star, 0.0
    cells = np.arange(j_e, N)
    lam_minus = (X2[cells + 1] - sbar[cells]) / (X2[cells + 1] - X2[cells])
    lam_plus = 1.0 - lam_minus
    faceN_bound = X_N**2 * most_out_of_face_n(eos, U_star, X_N, X_xi_N, float(rho[N - 1]) / float(lam_plus[-1]))

    F_frw = (alpha * w * X - X_xi) * X2
    face_scale = np.abs(F_frw) + np.abs(res.F)
    face_scale[faces] += X2[faces] * (
        (np.abs(A[faces]) + np.abs(vL[faces])) * rho_L[faces]
        + (np.abs(B[faces]) + np.abs(vR[faces])) * rho_R[faces]
        # the deviation form's dissipation term, D X^2 (delta rho^R - delta rho^L), carries eps D X^2 |delta rho|
        + np.fmax(Lp[faces], -Lm[faces]) * (np.abs(rho_L[faces] - 1.0) + np.abs(rho_R[faces] - 1.0))
    )
    face_scale[N] += faceN_bound + X_N**2 * abs(alpha * (1.0 + w) * U_star)
    if math.isfinite(v_star):  # the face value is formed in rho - 1: an absolute eps, carried at the face's own chord
        face_scale[N] += X_N**2 * abs(v_star) * abs(rho_hat - 1.0)

    # Layer 1: each cell's loss through its own two face values
    rho_minus, rho_plus = rho_R[cells], rho_L[cells + 1]
    out_coef = np.zeros(cells.size)
    below_N = cells + 1 < N
    out_coef[below_N] = X2[cells[below_N] + 1] * np.maximum(A[cells[below_N] + 1], 0.0)  # -F = -X^2 A rho^+_c + ...
    in_coef = np.zeros(cells.size)
    interior = cells > j_e  # the innermost retained cell's inner face is the origin (F_0 = 0) or the excision face
    in_coef[interior] = X2[cells[interior]] * np.maximum(B[cells[interior]], 0.0)  # face c: +F = -X^2 B rho^-_c + ...
    if j_e > 0:  # the excision face: F = X^2 v rho^-, a loss when v < 0
        in_coef[0] = X2[j_e] * max(-float(vR[j_e]), 0.0)
    loss = out_coef * rho_plus + in_coef * rho_minus
    loss[-1] += max(float(res.F[N]), 0.0)  # face N, read from the scheme: a loss when positive, else a gain
    loss_bound = (out_coef / lam_plus + in_coef / lam_minus) * rho[cells]
    loss_bound[-1] += faceN_bound
    E = case.state.E[cells]
    rate = res.rate.E[cells]
    cell_scale = face_scale[cells] + face_scale[cells + 1] + sigma * E + np.abs(geo.dV_xi[cells])
    return Audit(
        F=res.F,
        F_lemma=F_lemma,
        face_scale=face_scale,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho=rho[cells],
        lam_minus=lam_minus,
        E=E,
        dV=geo.dV[cells],
        rate=rate,
        loss=loss,
        loss_bound=loss_bound,
        cell_scale=cell_scale,
        faceN_bound=faceN_bound,
        dxi=courant_step(res, geo, lay, 1.0),
        defects=defects,
        A=A,
        B=B,
    )


# --- the checks: each returns what failed, empty if nothing did ---


def _where(case: Case, what: str, index: int, detail: str) -> str:
    return f"{case.family}/{case.seed} {what} {index}: {detail}"


def _flux_against_specification(sch: Scheme, case: Case, faces: range) -> list[str]:
    au = audit(sch, case)
    bad = [_where(case, "audit", -1, d) for d in au.defects if any(f"face {j}:" in d for j in faces)]
    for j in faces:
        err = abs(float(au.F[j] - au.F_lemma[j]))
        if not err <= FLUX_ROUNDOFF_EPS * EPS * float(au.face_scale[j]):
            bad.append(_where(case, "face", j, f"F = {au.F[j]:.6g}, specification {au.F_lemma[j]:.6g}"))
    return bad


def check_flux_interior(sch: Scheme, case: Case) -> list[str]:
    """The flux at every retained face below N (the excision face included) follows the lemma's specification: (C)
    with the chord-widened bounds (B). This pins the *bounds*; positivity is `check_no_drain` and `check_euler`.
    """
    return _flux_against_specification(sch, case, range(max(case.layout.j_e, 1), case.layout.N))


def check_flux_face_n(sch: Scheme, case: Case) -> list[str]:
    """The flux at face N follows the lemma's specification: the last cell's reconstructed value at its own lapse.
    This pins the *choice of face value*; that face N keeps the last cell positive is `check_face_n`, which the donor
    passes too. Split from the interior so that a bite changing only the bounds is told apart from wrong bounds.
    """
    return _flux_against_specification(sch, case, range(case.layout.N, case.layout.N + 1))


def check_face_n(sch: Scheme, case: Case) -> list[str]:
    """Face N: the last cell loses through it no more than an admissible face value of its own could carry out.

    With `F_N = X_N^2 rho_hat v*(rho_hat)` and `0 < rho_hat <= rho_(N-1) / lam^+`, the loss `max(F_N, 0)` is at most
    `X_N^2 sup_r max(r v*(r), 0)`, proportional to the last cell's own content; under inflow (`U* <= 0`) the bound is
    zero, so `F_N` must not be positive. A face value extrapolated from the neighbour breaks it.
    """
    au = audit(sch, case)
    N = case.layout.N
    F_N = float(au.F[N])
    if not F_N <= au.faceN_bound + FLUX_ROUNDOFF_EPS * EPS * float(au.face_scale[N]):
        return [_where(case, "face", N, f"F_N = {F_N:.6g} above the own-content bound {au.faceN_bound:.6g}")]
    return []


def _with_face_n(d: Derived, N: int, rho: float, delta_rho: float, eos: EquationOfState) -> Derived:
    """The derived fields with face N's `<rho>_N`, `<e^phi>_N` (and deviations) replaced by the pair of `rho`."""
    e, de = eos.lapse_and_deviation(np.array([rho]), np.array([delta_rho]))
    rho_f, drho_f, ephi_f, dephi_f = (
        np.array(v, copy=True) for v in (d.rho_f, d.delta_rho_f, d.ephi_f, d.delta_ephi_f)
    )
    rho_f[N], drho_f[N], ephi_f[N], dephi_f[N] = rho, delta_rho, float(e[0]), float(de[0])
    return dataclasses.replace(d, rho_f=rho_f, delta_rho_f=drho_f, ephi_f=ephi_f, delta_ephi_f=dephi_f)


def _velocity_row_n(d: Derived, sp: Speeds, deviation: State, case: Case, sch: Scheme) -> tuple[float, float]:
    """`OutgoingWave`'s velocity row at N (production strengths) read at `d`'s face N and `sp`'s drift there, and the
    size of its terms (for a round-off tolerance)."""
    N = case.layout.N
    frame = sch.frame(case.xi)
    geo, eos = frame.geo, case.eos
    alpha, w = float(eos.alpha), float(eos.w)
    X_N, c_s = float(geo.X[N]), frame.bg.c_s
    delta_DU_N = float(frame.w.velocity_gradient(deviation.U)[N])
    inputs = OuterInputs(
        xi=case.xi,
        X_N=X_N,
        X_xi_N=float(geo.X_xi[N]),
        U_N=float(case.state.U[N]),
        W=case.state.W,
        delta_U_N=float(d.delta_U[N]),
        delta_rho_N_1=float(d.delta_rho[N - 1]),
        rho_f_N=float(d.rho_f[N]),
        delta_rho_f_N=float(d.delta_rho_f[N]),
        ephi_f_N=float(d.ephi_f[N]),
        delta_ephi_f_N=float(d.delta_ephi_f[N]),
        mt_N=float(d.mt[N]),
        delta_m_N=float(d.delta_m[N]),
        drift_N=float(sp.drift[N]),
        delta_DU_N=delta_DU_N,
        dS_N=float(geo.dS[N]),
        c_s=c_s,
    )
    row = OutgoingWave(PRODUCTION_STRENGTHS).rows(inputs, eos).delta_dU_N
    _, u_minus = characteristic_pair(inputs.delta_U_N, inputs.delta_rho_N_1, X_N, c_s)
    lapse_mass = (
        abs(inputs.delta_ephi_f_N) * (abs(inputs.mt_N) + 3.0 * w * inputs.rho_f_N)
        + abs(inputs.delta_m_N)
        + 3.0 * w * abs(inputs.delta_rho_f_N)
    )
    size = (  # the row's four terms in size: expansion, gravity, advection, penalty
        (1.0 - alpha) * X_N * abs(inputs.delta_U_N)
        + 0.5 * alpha * X_N * lapse_mass
        + abs(inputs.drift_N) * (1.0 + abs(delta_DU_N))
        + PRODUCTION_STRENGTHS.tau_u * c_s * X_N**2 / inputs.dS_N * abs(u_minus - inputs.W)
    )
    return row, size


def check_pair_n(sch: Scheme, case: Case) -> list[str]:
    """Face N has one state (lemma.md Sec. 5.4): the pair `(rho_hat_N, e^phi(rho_hat_N))`, `rho_hat_N` the last cell's
    reconstructed value at face N (as `check_flux_face_n` reads it, `rho^L_N`). From production outputs only:

      (a) the stage's derived `<rho>_N` and `<e^phi>_N` (with their deviations) ARE that pair;
      (b) its face-N sound speed `a_N` and Courant speed `Lambda_N` are what `pbh.equations.speeds` forms from the
          derived fields with face N set to the pair;
      (c) its deviation rate of `U_N` is `pbh.outer.OutgoingWave`'s velocity row read at the pair (and at (b)'s drift).

    Today face N is the extrapolation `<f>_N = 3/2 f_(N-1) - 1/2 f_(N-2)` of both the density and the lapse, which can
    turn non-positive (lemma.md Sec. 5.4, P29). Tolerances: the pair is known to an absolute `eps (1 + rho_hat)`, so
    each comparison allows the change `PAIR_ROUNDOFF_EPS` such `eps` in the density make, plus that many `eps` of the
    quantity's own size. The SAT closure only: the held face of the pinned map is outside the lemma (Sec. 1.5).
    """
    if case.closure != "sat":
        return []
    N = sch.layout.N
    res = sch.evaluate(case.xi, sch.layout.pack(case.state))
    k = res.kernels
    if k is None:  # not an assert: a strict xfail must be satisfied only by the check's own assertions
        raise RuntimeError("the kernels are off; the face-N pair is the reconstruction's value at face N")
    return _pair_n_holds(sch, case, res, float(k.rho_L[N]), float(k.delta_rho_L[N]))


def face_n_value_kernels_off(sch: Scheme, case: Case, d: Derived) -> tuple[float, float]:
    """`(rho_hat_N, rho_hat_N - 1)` formed here, independently of any reconstruction the scheme reports: the last
    cell's one-sided `s`-slope from cells `N-2` and `N-1`, `(delta rho_(N-1) - delta rho_(N-2)) / dS_(N-1)`,
    theta-limited by (T1) at the scheme's theta (both face values of cell `N-1` at least `theta rho_(N-1)`),
    evaluated at face N (lemma.md Sec. 5.4 reason 5: the pair is defined without the kernels)."""
    N = case.layout.N
    geo = sch.frame(case.xi).geo
    d1, d2 = float(d.delta_rho[N - 1]), float(d.delta_rho[N - 2])
    slope = (d1 - d2) / float(geo.dS[N - 1])
    sbar = float(geo.sbar[N - 1])
    off_out = slope * (float(geo.X[N]) ** 2 - sbar)
    off_in = slope * (float(geo.X[N - 1]) ** 2 - sbar)
    rho1 = 1.0 + d1
    drop = -min(off_in, off_out, 0.0)  # the larger drop of a face value below the mean
    need = (1.0 - sch.settings.theta) * rho1
    t = need / drop if drop > need else 1.0
    return rho1 + t * off_out, d1 + t * off_out


def check_pair_n_kernels_off(sch: Scheme, case: Case) -> list[str]:
    """`check_pair_n` for the centred base scheme (the kernels off, `CENTRED_SCHEME`: the test switch), which has no
    reconstruction to report, so `rho_hat_N` is formed here by `face_n_value_kernels_off` from the last two cells,
    never read from `res.kernels`. The scheme under test is run with its kernels switched off (the same map, layout
    and outer closure), and (a)-(c) of `check_pair_n` are asserted against that pair. The SAT closure only.
    """
    if case.closure != "sat":
        return []
    centred = dataclasses.replace(sch, settings=CENTRED_SCHEME)
    res = centred.evaluate(case.xi, centred.layout.pack(case.state))
    if res.kernels is not None:  # not an assert: a strict xfail must be satisfied only by the check's own assertions
        raise RuntimeError("the kernels are on; this check is the kernels-off path")
    rho_hat, drho_hat = face_n_value_kernels_off(centred, case, res.derived)
    return _pair_n_holds(centred, case, res, rho_hat, drho_hat)


def _pair_n_holds(sch: Scheme, case: Case, res: DerivsResult, rho_hat: float, drho_hat: float) -> list[str]:
    """(a)-(c) of `check_pair_n` for the evaluation `res` of `sch` on the case, against the pair of `rho_hat`."""
    lay = sch.layout
    N, eos = lay.N, case.eos
    if not rho_hat > 0.0:
        return [_where(case, "face", N, f"the face value {rho_hat:.3g} is not positive")]
    geo = sch.frame(case.xi).geo
    d = res.derived
    deviation = deviation_from_frw(case.state, geo, lay.j_e)
    shift = PAIR_ROUNDOFF_EPS * EPS * (1.0 + rho_hat)  # the pair's own round-off, as an absolute density
    pair = _with_face_n(d, N, rho_hat, drho_hat, eos)
    near = _with_face_n(d, N, rho_hat + shift, drho_hat + shift, eos)
    bad: list[str] = []

    def close(value: float, want: float, want_shifted: float) -> bool:
        return abs(value - want) <= abs(want_shifted - want) + PAIR_ROUNDOFF_EPS * EPS * abs(want)

    # (a) the derived face state
    if not (abs(float(d.rho_f[N]) - rho_hat) <= shift and abs(float(d.delta_rho_f[N]) - drho_hat) <= shift):
        bad.append(_where(case, "face", N, f"<rho>_N = {d.rho_f[N]:.6g}, the pair's {rho_hat:.6g}"))
    for name in ("ephi_f", "delta_ephi_f"):
        got, want, shifted = (float(getattr(x, name)[N]) for x in (d, pair, near))
        if not close(got, want, shifted) or not abs(got - want) <= abs(shifted - want) + shift:
            bad.append(_where(case, "face", N, f"{name}[N] = {got:.6g}, the pair's {want:.6g}"))
    # (b) the face-N speeds, formed by production's own speeds() from the pair
    sp_pair = speeds(pair, deviation, geo, eos, lay.faces)
    sp_near = speeds(near, deviation, geo, eos, lay.faces)
    for name in ("a", "Lam"):
        got, want, shifted = (float(getattr(x, name)[N]) for x in (res.speeds, sp_pair, sp_near))
        if not close(got, want, shifted):
            bad.append(_where(case, "face", N, f"{name}_N = {got:.6g}, speeds() at the pair gives {want:.6g}"))
    # (c) the velocity row at N, production's OutgoingWave rows read at the pair
    want_row, size = _velocity_row_n(pair, sp_pair, deviation, case, sch)
    near_row, _ = _velocity_row_n(near, sp_near, deviation, case, sch)
    got_row = float(res.deviation_rate.U[N])
    if not abs(got_row - want_row) <= abs(near_row - want_row) + PAIR_ROUNDOFF_EPS * EPS * size:
        bad.append(_where(case, "face", N, f"d_xi delta U_N = {got_row:.6g}, OutgoingWave at the pair {want_row:.6g}"))
    return bad


def check_theta_faces(sch: Scheme, case: Case) -> list[str]:
    """Both reconstructed face values of every retained cell are at least `THETA_MIN` times its density, to round-off.

    The bound holds to `O(eps)` absolute (the face values are formed in `rho - 1`), which is `eps / rho` relative: at
    `rho = 3e-9` a face value the limiter puts exactly at `theta rho_c` can come out `1e-8` below it, relatively.
    """
    au = audit(sch, case)
    low = np.minimum(au.rho_minus, au.rho_plus)
    ok = low >= THETA_MIN * au.rho - THETA_ROUNDOFF_EPS * EPS * (1.0 + au.rho)
    return [
        _where(case, "cell", case.layout.j_e + int(i), f"face value {low[i] / au.rho[i]:.3g} of the cell's density")
        for i in np.flatnonzero(~ok)
    ]


def check_zhang_shu(sch: Scheme, case: Case) -> list[str]:
    """(ZS): the face values are non-negative, their weighted mean is the cell's density, and so `L_c` obeys (K)."""
    au = audit(sch, case)
    mean = au.lam_minus * au.rho_minus + (1.0 - au.lam_minus) * au.rho_plus
    bad: list[str] = []
    for i in range(au.rho.size):
        c = case.layout.j_e + i
        if not min(au.rho_minus[i], au.rho_plus[i]) >= 0.0:
            bad.append(_where(case, "cell", c, f"negative face value {min(au.rho_minus[i], au.rho_plus[i]):.3g}"))
        if not abs(mean[i] - au.rho[i]) <= ZS_ROUNDOFF_EPS * EPS * (1.0 + au.rho[i]):
            bad.append(_where(case, "cell", c, f"mean of the face values {mean[i]:.6g} against {au.rho[i]:.6g}"))
        if not au.loss[i] <= au.loss_bound[i] * (1.0 + 1e-9) + 1e-14 * au.cell_scale[i]:
            bad.append(_where(case, "cell", c, f"loss {au.loss[i]:.3g} above its (ZS) bound {au.loss_bound[i]:.3g}"))
    return bad


def check_no_drain(sch: Scheme, case: Case) -> list[str]:
    """Layer 1: `rate_c + L_c - sigma E_c >= 0`: whatever a cell loses beyond its own face values is a gain.

    Two tolerances, both of which must hold: round-off, `GAIN_ROUNDOFF_EPS eps` times the size of the terms; and
    content, a drain over one Courant step (`C = 1`) below `DRAIN_PER_STEP` of the cell's content, so that no neighbour
    can empty a near-empty cell in one step while hiding under the round-off of the dense cells beside it.
    """
    au = audit(sch, case)
    gain = au.rate + au.loss - case.eos.energy_source_rate * au.E
    roundoff = CONTENT_ROUNDOFF_EPS * EPS * (au.dV + au.dxi * au.cell_scale)
    ok = (gain >= -GAIN_ROUNDOFF_EPS * EPS * au.cell_scale) & (-gain * au.dxi <= DRAIN_PER_STEP * au.E + roundoff)
    return [
        _where(case, "cell", case.layout.j_e + int(i), f"drained by its neighbours at {-gain[i] / au.E[i]:.3g} E")
        for i in np.flatnonzero(~ok)
    ]


def check_euler(sch: Scheme, case: Case) -> list[str]:
    """Layer 2: a forward-Euler step at `dxi = 0.9 / max(K - sigma)` keeps every retained content positive.

    The lemma promises more, `E' >= (1 - dxi (K - sigma)) E >= 0.1 E`, but `E' - (1 - dxi (K - sigma)) E` is
    `dxi` times the gain of `check_no_drain`, so that part is the same assertion; this one asks only for what the step
    is for, a positive content, which a drain can deny only by exceeding the content in one certificate step.
    """
    au = audit(sch, case)
    excess = au.loss / au.E - case.eos.energy_source_rate
    top = float(np.max(excess))
    dxi = CERTIFICATE_FRACTION / top if top > 0.0 else 1.0
    E_new = au.E + dxi * au.rate
    return [
        _where(case, "cell", case.layout.j_e + int(i), f"E'/E = {E_new[i] / au.E[i]:.4g} at dxi = {dxi:.3g}")
        for i in np.flatnonzero(~(E_new > 0.0))
    ]


type Check = Callable[[Scheme, Case], list[str]]
CHECKS: dict[str, Check] = {
    "flux_int": check_flux_interior,
    "flux_N": check_flux_face_n,
    "pair_N": check_pair_n,
    "pair_N_off": check_pair_n_kernels_off,
    "face_n": check_face_n,
    "theta_faces": check_theta_faces,
    "zhang_shu": check_zhang_shu,
    "no_drain": check_no_drain,
    "euler": check_euler,
}


def failures(
    check: Check, family: str, scheme_for: Callable[[Case], Scheme] = production_scheme, w: str = RAD
) -> list[str]:
    """Every failure of the check over the family's seeds at the equation of state `EOSES[w]`."""
    found: list[str] = []
    for seed in SEEDS:
        case = violent_case(family, seed, w)
        found += check(scheme_for(case), case)
    return found


# --- the tests that hold today: the generator, the transcription of the lemma, the excision face ---

EVERY_CASE = [pytest.param(w, f, id=f"{w}-{f}") for w in W_CASES for f in FAMILIES]


@pytest.mark.parametrize(("w", "family"), EVERY_CASE)
def test_the_generated_states_are_violent_and_admissible(w: str, family: str):
    eos = EOSES[w]
    deepest = 1.0
    for seed in SEEDS:
        case = violent_case(family, seed, w)
        sch = production_scheme(case)
        res = sch.evaluate(case.xi, sch.layout.pack(case.state))  # raises NotHyperbolicError if not admissible
        rho = res.derived.rho[case.layout.cells]
        assert np.min(rho) > 0.0
        assert np.max(rho) / np.min(rho) > 1e3  # a near-empty cell somewhere beside a dense one
        jumps = np.abs(np.diff(res.derived.delta_U[max(case.layout.j_e, 1) :]))
        assert np.max(jumps) > 0.5  # an order-one velocity jump between neighbouring faces
        deepest = min(deepest, float(np.min(rho)))
    assert deepest < 3e-12  # the deep seeds reach towards the 5e-13 abort threshold
    if family == "excised":
        chord_out = today_open = 0
        for seed in SEEDS:
            case = violent_case(family, seed, w)
            sch = production_scheme(case)
            res = sch.evaluate(case.xi, case.layout.pack(case.state))
            assert res.kernels is not None
            j_e = case.layout.j_e
            assert j_e > 0
            trapped = case.state.U[j_e] + math.sqrt(res.derived.Gammabar2[j_e]) < 0.0  # the finder's trapping
            Theta_plus_a = float(res.speeds.Theta[j_e] + res.speeds.a[j_e])
            assert trapped == (case.kind != "open")
            assert (Theta_plus_a < 0.0) == trapped  # trapped faces are acoustically trapped too; open ones are not
            geo = sch.frame(case.xi).geo
            e = np.array([j_e])
            v_e = chord(
                eos, res.kernels.rho_R[e], case.state.U[e], geo.X[e], geo.X_xi[e], res.kernels.q[e] / res.derived.rho[e]
            )
            chord_out += int(trapped and float(v_e[0]) > 0.0)  # the chord bounds' Lambda+ > 0 at a trapped face
            today_open += int(max(Theta_plus_a, 0.0) > 0.0)  # today's Lambda+ > 0 at the face
        # the excision rule rho^L := rho^R is visible to a flux only where Lambda+ > 0 at j_e: both kinds must occur
        assert chord_out >= 1
        assert today_open >= 1
    if family == "superhorizon":  # most of the grid beyond the chord crossover X_c = e^((1-alpha) xi) / sqrt(w)
        case = violent_case(family, 0, w)
        X_max = float(case.map.radii(case.xi, 40)[0][-1])
        assert float(eos.sqrt_w) * X_max > 10.0 * math.exp((1 - float(eos.alpha)) * case.xi)
    if family in ("blend", "pinned"):
        case = violent_case(family, 0, w)
        assert np.max(np.abs(case.map.radii(case.xi, 40)[1])) > 0.1  # the map moves
    if family == "outer_inflow":
        case = violent_case(family, 0, w)
        assert case.state.U[-1] < 0.0  # the flow through the outer face is inward


def test_an_inadmissible_state_is_refused_not_audited():
    # The generator's Gammabar^2 guard is what keeps the stage evaluable: without it the mass of the dense cells
    # makes Gammabar^2 negative, and the stage refuses the state rather than returning a rate to audit.
    case = violent_case("void", 0)
    U = case.state.U.copy()
    U[1:] = 0.0
    bad = Case(case.family, case.seed, case.map, case.layout, case.xi, State(E=case.state.E * 1e3, U=U, W=0.0), "sat")
    with pytest.raises(NotHyperbolicError):
        audit(production_scheme(bad), bad)


@pytest.mark.parametrize(("w", "family"), EVERY_CASE)
def test_the_coefficient_identity_reproduces_todays_flux_with_its_own_bounds(w: str, family: str):
    # (C) with the scheme's own bounds Theta +- a returns its flux at every interior and excision face, whatever the
    # sign of A and B: the chords, the density-weighted viscous work and the algebra here are transcribed right, so
    # the failures below are the bounds', not the audit's. This anchor is TODAY'S: under the chord bounds it fails on
    # every family (the audit's "acoustic" bounds are then not the scheme's), and the scheme-change bite must delete
    # it, the anchor passing to test_every_interior_flux_follows_the_lemmas_flux_specification (module docstring).
    for seed in SEEDS:
        case = violent_case(family, seed, w)
        au = audit(production_scheme(case), case, bounds="acoustic")
        j = np.arange(max(case.layout.j_e, 1), case.layout.N)
        assert np.all(np.abs(au.F[j] - au.F_lemma[j]) <= FLUX_ROUNDOFF_EPS * EPS * au.face_scale[j]), f"{family}/{seed}"


@pytest.mark.parametrize("w", list(W_CASES))
def test_the_excision_face_carries_only_the_first_cells_own_content(w: str):
    # rho^L := rho^R and one q / rho on both sides make v^L = v^R, so F_je = X^2 v rho^- whatever the bounds: the
    # excision face is inside the lemma as the scheme stands. The rule matters only where Lambda+ > 0 at the face
    # (with Lambda+ = 0 the HLL flux is F(rho^R) whatever rho^L is): today that is the open seeds; with the chord
    # bounds also the trapped face whose chord points outward (the generator asserts both occur). Setting
    # rho^L_je to the cell mean makes this test fail on the open seeds (lemma.md Sec. 5.2).
    for seed in SEEDS:
        case = violent_case("excised", seed, w)
        sch = production_scheme(case)
        res = sch.evaluate(case.xi, sch.layout.pack(case.state))
        assert res.kernels is not None
        j_e = case.layout.j_e
        geo = sch.frame(case.xi).geo
        e = np.array([j_e])
        q_over_rho = res.kernels.q[e] / res.derived.rho[e]
        v = chord(case.eos, res.kernels.rho_R[e], case.state.U[e], geo.X[e], geo.X_xi[e], q_over_rho)[0]
        assert res.F[j_e] == pytest.approx(float(geo.X[j_e] ** 2 * v * res.kernels.rho_R[j_e]), rel=1e-10)


# --- the lemma against today's scheme: strict xfails, each naming what the scheme lacks ---

NO_CHORD = "the HLL bounds are Theta +- a, which do not bracket the one-sided chord speeds (chord bounds missing)"


def _lemma_params(missing: dict[tuple[str, str], str]) -> list[object]:
    """Every (w, family), each marked strict xfail with the ingredient today's scheme lacks for it, if it fails.

    Which cases fail was measured (proptest_prototype.py prints the table, "production", at each w, and checks that
    these markers are exactly today's failures). A case left unmarked passes today: at w = 1/3, for the drain and the
    step, the void and the shock, where the face-averaged lapse in `Theta +- a` widens the bracket around an emptying
    cell by itself (lemma.md Section 2.4) or the converging flow feeds the empty cells, and the excised and blend
    states for the step, whose drains are smaller than a content. At w = 1 that partial self-protection does not
    suffice anywhere: the drain and the step fail on every family.
    Only an `AssertionError` counts as the expected failure: an audit that raises anything else is a defect of the
    test, and the audit's own preconditions raise `RuntimeError` for that reason.
    """
    return [
        pytest.param(
            w, f, id=f"{w}-{f}", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=missing[(w, f)])
        )
        if (w, f) in missing
        else pytest.param(w, f, id=f"{w}-{f}")
        for w in W_CASES
        for f in FAMILIES
    ]


def _cases(families: str, ws: tuple[str, ...] = tuple(W_CASES)) -> list[tuple[str, str]]:
    return [(w, f) for w in ws for f in families.split()]


ALL = " ".join(FAMILIES)
SAT = " ".join(f for f in FAMILIES if f != "pinned")
FLUX_INTERIOR: dict[tuple[str, str], str] = dict.fromkeys(_cases(ALL), NO_CHORD)
# The theta-limiter's checks carry no markers: the scheme has the theta-limiter and the face-N pair.
FLUX_FACE_N: dict[tuple[str, str], str] = {}
PAIR_N: dict[tuple[str, str], str] = {}
PAIR_N_OFF: dict[tuple[str, str], str] = {}
THETA_FACES: dict[tuple[str, str], str] = {}
FACE_N: dict[tuple[str, str], str] = {}
ZHANG_SHU: dict[tuple[str, str], str] = {}
DRAINS: dict[tuple[str, str], str] = dict.fromkeys(
    _cases("superhorizon stretched excised blend pinned outer_inflow", (RAD,)) + _cases(ALL, ("w=1",)), NO_CHORD
)
NEGATIVE: dict[tuple[str, str], str] = dict.fromkeys(
    _cases("superhorizon stretched pinned outer_inflow", (RAD,)) + _cases(ALL, ("w=1",)), NO_CHORD
)
#: The strict xfails of each check, by the check's key in CHECKS (proptest_prototype.py predicts from these which
#: markers a partial scheme change would flip).
MARKERS: dict[str, dict[tuple[str, str], str]] = {
    "flux_int": FLUX_INTERIOR,
    "flux_N": FLUX_FACE_N,
    "pair_N": PAIR_N,
    "pair_N_off": PAIR_N_OFF,
    "face_n": FACE_N,
    "theta_faces": THETA_FACES,
    "zhang_shu": ZHANG_SHU,
    "no_drain": DRAINS,
    "euler": NEGATIVE,
}

#: The checks each ingredient of the lemma implements. Once the scheme has the ingredient, EVERY marker of these checks
#: must be deleted wholesale, not only those that XPASS: a strict xfail whose case still fails stays green, so a marker
#: kept after the bite would hide a defect of the bite (a wrong face value at the excision face, acoustic bounds left on
#: some faces, bounds without the viscous work, a theta below `THETA_MIN`, end cells not theta-limited, a broken mean in
#: the near-empty cells all fail mostly or only behind today's markers). The theta-limiter's bite carries the outer
#: closure with it (one theta, one reconstruction, lemma.md Sec. 6), so it owns the face-N checks too, the face-N pair
#: of the derived fields, the speeds and the velocity row among them (lemma.md Sec. 5.4).
IMPLEMENTS: dict[str, tuple[str, ...]] = {
    "chord bounds": ("flux_int", "no_drain", "euler"),
    "theta-limiter": ("theta_faces", "zhang_shu", "face_n", "flux_N", "pair_N", "pair_N_off"),
}
#: A face value below this fraction of its cell's density is not theta-limited by any admissible theta; mc puts face
#: values at a near-empty neighbour's level (1e-9 to 1e-4 of the cell in the void states), so today's scheme has them.
THETA_DETECT = 1e-3


def installed_ingredients(scheme_for: Callable[[Case], Scheme] | None = None) -> set[str]:
    """Which ingredients of the lemma the scheme has, read from its behaviour, since the scheme reports no mode flag.

    The chord bounds: on the superhorizon states, some interior face where the chord-bound and the acoustic
    specifications differ by more than a thousand times the flux tolerance carries the chord-bound flux. The
    theta-limiter: on the void states, no interior cell (away from the end cells) has a face value below
    `THETA_DETECT` times its density. Each reads the ingredient's presence anywhere, so a bite that has it on some faces
    or cells only is still detected, and the guard below then exposes the rest. The scheme defaults to
    `production_scheme`, looked up when called.
    """
    build = production_scheme if scheme_for is None else scheme_for
    found: set[str] = set()
    for seed in SEEDS:
        case = violent_case("superhorizon", seed)
        sch = build(case)
        chord_au, acoustic_au = audit(sch, case), audit(sch, case, bounds="acoustic")
        j = np.arange(1, case.layout.N)
        tol = FLUX_ROUNDOFF_EPS * EPS * chord_au.face_scale[j]
        differ = np.abs(chord_au.F_lemma[j] - acoustic_au.F_lemma[j]) > 1e3 * tol
        follows = np.abs(chord_au.F[j] - chord_au.F_lemma[j]) <= tol
        if bool(np.any(differ & follows)):
            found.add("chord bounds")
    lowest = math.inf
    for seed in SEEDS:
        case = violent_case("void", seed)
        au = audit(build(case), case)
        inner = slice(1, au.rho.size - 1)
        lowest = min(lowest, float(np.min(np.minimum(au.rho_minus[inner], au.rho_plus[inner]) / au.rho[inner])))
    if lowest >= THETA_DETECT:
        found.add("theta-limiter")
    return found


def test_no_marker_outlives_the_ingredient_that_implements_its_check():
    # The strict xfails force an edit only when a marked case XPASSes; this forces the rest: once the scheme has an
    # ingredient, the markers of every check it implements must be gone, so that any case still failing is red.
    # Today neither ingredient is detected and the test holds trivially; proptest_prototype.py shows that it fires
    # with the prototype installed while the markers remain, and that the detection finds each ingredient on its own.
    stale = {
        check: len(MARKERS[check])
        for ingredient in sorted(installed_ingredients())
        for check in IMPLEMENTS[ingredient]
        if MARKERS[check]
    }
    assert not stale, f"markers left on checks the scheme now implements (delete them wholesale): {stale}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(FLUX_INTERIOR))
def test_every_interior_flux_follows_the_lemmas_flux_specification(w: str, family: str):
    # Pins the bounds (chord-widened) at every retained face below N; that the resulting flux keeps every cell
    # positive is the business of the no-drain and forward-Euler tests below.
    bad = failures(check_flux_interior, family, w=w)
    assert not bad, f"{len(bad)} faces, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(FLUX_FACE_N))
def test_the_face_n_flux_follows_the_lemmas_flux_specification(w: str, family: str):
    # Pins the face value at N (the last cell's reconstruction at its own lapse); positivity there is the face-N test.
    bad = failures(check_flux_face_n, family, w=w)
    assert not bad, f"{len(bad)} faces, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(PAIR_N))
def test_face_n_has_one_state_the_pair_of_its_reconstructed_value(w: str, family: str):
    # The derived <rho>_N, <e^phi>_N, the face-N speeds and the velocity row at N all read the pair (rho_hat_N,
    # e^phi(rho_hat_N)) that F_N carries (lemma.md Sec. 5.4); today they read the extrapolations, which can turn
    # negative.
    bad = failures(check_pair_n, family, w=w)
    assert not bad, f"{len(bad)} faces, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(PAIR_N_OFF))
def test_with_the_kernels_off_face_n_has_one_state_the_pair_of_the_last_cells_slope(w: str, family: str):
    # The same contract on the centred base scheme (the test switch), which has no reconstruction: the pair is formed
    # here from the theta-limited one-sided s-slope of cells N-2 and N-1 (lemma.md Sec. 5.4 reason 5).
    bad = failures(check_pair_n_kernels_off, family, w=w)
    assert not bad, f"{len(bad)} faces, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(THETA_FACES))
def test_every_face_value_is_at_least_theta_times_its_cell(w: str, family: str):
    bad = failures(check_theta_faces, family, w=w)
    assert not bad, f"{len(bad)} cells, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(FACE_N))
def test_the_last_cell_loses_through_face_n_only_what_its_own_face_value_carries(w: str, family: str):
    bad = failures(check_face_n, family, w=w)
    assert not bad, f"{len(bad)} faces, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(ZHANG_SHU))
def test_the_face_values_keep_the_mean_and_bound_the_loss(w: str, family: str):
    bad = failures(check_zhang_shu, family, w=w)
    assert not bad, f"{len(bad)} cells, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(DRAINS))
def test_no_cell_is_drained_by_its_neighbours(w: str, family: str):
    bad = failures(check_no_drain, family, w=w)
    assert not bad, f"{len(bad)} cells, e.g. {bad[:3]}"


@pytest.mark.parametrize(("w", "family"), _lemma_params(NEGATIVE))
def test_a_forward_euler_step_at_the_certificate_step_keeps_every_cell_positive(w: str, family: str):
    bad = failures(check_euler, family, w=w)
    assert not bad, f"{len(bad)} cells, e.g. {bad[:3]}"
