"""The monitors of a run before formation: the scalars recorded every step (output specification, Section 3).

Most runs never have their monitors read, so the record has two tiers. Every step records what is free or one reduction
each: the step, the smallest density and `Gammabar^2` of the state arrived at, the total mass, the Runge-Kutta-weighted
outer flux with its integral and the bookkeeping residual, the central density and the outer boundary's scalars; about
one per cent of a step. At every snapshot, where someone looking at the fields will want them, and on every step if the
configuration's `monitor_every_step` is on, the row is full: the locations of the minima, the Courant cell, the
theta-limiter's bind count, the far zone, the boundary energy, the grid scale, the steepness probe, and the
under-resolution monitors of the criticality study, about a fifth of a step. On the other steps those columns hold NaN,
or `-1` for counts.

`StageFluxes` is what each stage of a step contributes, three scalars; `MonitoredStep` is the row of the step table,
`StepRow` extended with both tiers; `monitor_step` fills it.

The under-resolution monitors are the ones the criticality design asks for on a single run, since a threshold
decision has no internal error signal of its own: the artificial viscous pressure against the physical pressure
in the core, the limiter's clipping and whether it reaches the core, the reconstruction jump at the faces, the
number of cells inside the half-central-density radius, and the grid-scale content. They flag under-resolution;
they cannot certify its absence, which is what the refinement pair of the error analysis is for.

Some quantities are defined in terms of a spectrum on a uniform grid and are approximated here on any grid:
the grid-scale content is the fraction of a field's root-mean-square deviation that sits in its alternating
component `f_c - (f_{c-1} + f_{c+1}) / 2`, the sawtooth detector of Section 7.4.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from pbh.derived import Derived
from pbh.eos import Background, EquationOfState
from pbh.equations import DerivsResult
from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.outer import characteristic_pair
from pbh.output import StepRow
from pbh.state import State
from pbh.types import FloatArray

# --- what one stage contributes ---


@dataclass(frozen=True)
class StageFluxes:
    """The scalars of a stage that the step's bookkeeping needs; free to collect.

    The total mass obeys `d_xi M_total = (2 - 3 alpha) M_total - 3 F_N` exactly. The outer face is static, so the FRW
    parts of both sides cancel identically, `(2 - 3 alpha) X_N^3 = 3 alpha w X_N^3`, and the bookkeeping is checked in
    the deviations, `d_xi delta M_total = (2 - 3 alpha) delta M_total - 3 delta F_N`, where it is limited by the
    rounding of the deviation rather than of `M_total` itself.

    Attributes:
        F_N: The outer flux; `F_je` the flux through the excision face (`0` unexcised).
        M_total: The total mass in the domain, `M_N = M_e + 3 sum E_c`.
        delta_F_N: The outer flux less its FRW value.
        delta_M_total: The total mass less its FRW value `X_N^3`, the cumulative sum of the deviations.
    """

    F_N: float
    F_je: float
    M_total: float
    delta_F_N: float
    delta_M_total: float

    @classmethod
    def of(cls, result: DerivsResult, layout: Layout) -> StageFluxes:
        """Collect the stage's fluxes and total mass."""
        N, j_e = layout.N, layout.j_e
        d = result.derived
        return cls(
            F_N=float(result.F[N]),
            F_je=float(result.F[j_e]) if j_e > 0 else 0.0,
            M_total=float(d.M[N]),
            delta_F_N=float(result.delta_F[N]),
            delta_M_total=float(d.delta_M[N]),
        )


# --- the row of the step table ---


@dataclass(frozen=True)
class MonitoredStep(StepRow):
    """The step record: `StepRow` and the monitors of Section 3 of the output specification, pre-formation.

    The first tier is recorded every step; the second at snapshots, or every step on request, NaN or `-1` otherwise.
    """

    # --- every step ---
    rho_min: float
    Gammabar2_min: float
    """The smallest `Gammabar_j^2 e^(-2 (1 - alpha) xi)`, the margin of Section 7.6."""
    M_total: float
    F_N: float
    """The outer flux, Runge-Kutta-weighted over the step."""
    F_N_integral: float
    """The running integral of the outer flux, `int F_N dxi`."""
    bookkeeping_residual: float
    """`M_total` after the step minus what the weighted stage rates predict, relative to `M_total`: round-off. Formed
    in the deviations (`StageFluxes`), so it is the rounding of the deviation, not of `M_total`."""
    rho_0: float
    W: float
    penalty: float
    """The residual `u_- - W` of the outer condition."""
    companion: float
    """RK4's third-order companion estimate, `dxi / 6` times the largest change of the rate across the step, relative
    to the state's scale; logged, never used for control."""
    # --- at snapshots, or every step with monitor_every_step ---
    rho_min_cell: int
    Gammabar2_min_face: int
    theta_binds: int
    courant_ratio: float
    courant_cell: int
    Theta_at_courant: float
    a_at_courant: float
    far_zone_delta_rho: float
    far_zone_delta_U: float
    u_plus: float
    u_minus: float
    delta_m_N: float
    delta_rho_N_1: float
    boundary_energy: float
    """The augmented energy of Section 7.5: the norm eq:num:norm with face `N`, plus `E_b`; NaN off radiation."""
    delta_U_1: float
    odd_even_inner: float
    """The alternating amplitude of the density over cells 0-3, relative to the central density."""
    grid_scale_cells: float
    grid_scale_faces: float
    steepest_log_slope: float
    steepest_X: float
    largest_cell_jump: float
    largest_cell_jump_outside_3: float
    largest_four_cell_ratio_outside_5: float
    supersonic_faces: int
    supersonic_outer_X: float
    core_cells: int
    """Cells between the origin and the radius where the density has fallen to half its central value."""
    viscous_over_pressure_core: float
    """`max |q_c| / (w rho_c)` over the core: the artificial against the physical pressure."""
    clipped_cells: int
    """Cells whose limited density slope differs from the centred one."""
    clipped_in_core: int
    reconstruction_jump: float
    """`max |rho_R - rho_L| / <rho>` over the faces."""


@dataclass(frozen=True)
class StepInputs:
    """Everything `monitor_step` needs about one completed step.

    Attributes:
        step, xi, dxi, limit: The step's identification, as in `StepRow`; `xi` the time arrived at.
        state, geo, bg, result: The state arrived at, its geometry and background, and the stage result there.
        stages: The fluxes of the step's stages, in order, and `weights` the tableau's `b`.
        delta_M_total_before: The total mass's deviation from FRW before the step, for the bookkeeping residual.
        F_N_integral_before: The running flux integral before the step.
        rate_change: The largest change of the packed rate from the step's last stage to the evaluation at the new
            state, relative to the state's scale; `dxi / 6` times it is the companion estimate.
        far_zone_from: The radius beyond which the initial data were FRW.
    """

    step: int
    xi: float
    dxi: float
    limit: str
    state: State
    geo: Geometry
    bg: Background
    result: DerivsResult
    stages: list[StageFluxes]
    weights: tuple[float, ...]
    delta_M_total_before: float
    F_N_integral_before: float
    rate_change: float
    far_zone_from: float


def monitor_step(inputs: StepInputs, eos: EquationOfState, layout: Layout, full: bool = True) -> MonitoredStep:
    """Fill the step record from a completed step: the first tier always, the second when `full`."""
    i = inputs
    N, j_e = layout.N, layout.j_e
    cells, faces = layout.cells, layout.faces
    geo, bg, d, state = i.geo, i.bg, i.result.derived, i.state
    X = geo.X[: N + 1]

    # the first tier
    end = StageFluxes.of(i.result, layout)
    F_N = sum(b * s.F_N for b, s in zip(i.weights, i.stages, strict=True))
    predicted = i.delta_M_total_before + i.dxi * sum(
        b * (eos.energy_source_rate * s.delta_M_total - 3.0 * s.delta_F_N)
        for b, s in zip(i.weights, i.stages, strict=True)
    )
    u_plus, u_minus = characteristic_pair(float(d.delta_U[N]), float(d.delta_rho[N - 1]), float(X[N]), bg.c_s)
    every_step = {
        "rho_min": float(np.min(d.rho[cells])),
        "Gammabar2_min": float(np.min(d.Gammabar2[faces])) / bg.Gammabar2,
        "M_total": end.M_total,
        "F_N": F_N,
        "F_N_integral": i.F_N_integral_before + i.dxi * F_N,
        "bookkeeping_residual": abs(end.delta_M_total - predicted) / end.M_total,
        "rho_0": float(d.rho[j_e]),
        "W": state.W,
        "penalty": u_minus - state.W,
        "companion": i.dxi / 6.0 * i.rate_change,
    }
    if not full:
        return MonitoredStep(i.step, i.xi, i.dxi, i.limit, **every_step, **UNSET_COLUMNS)

    # the second tier
    diag = full_diagnostics(i, eos, layout, u_plus, u_minus, every_step["rho_0"])
    return MonitoredStep(i.step, i.xi, i.dxi, i.limit, **every_step, **dataclasses.asdict(diag))


def full_diagnostics(
    i: StepInputs,
    eos: EquationOfState,
    layout: Layout,
    u_plus: float,
    u_minus: float,
    rho_0: float,
) -> Diagnostics:
    """The second tier: everything with an index, an array or a norm in it."""
    N, j_e = layout.N, layout.j_e
    cells, faces = layout.cells, layout.faces
    geo, bg, d, state, sp = i.geo, i.bg, i.result.derived, i.state, i.result.speeds
    X = geo.X[: N + 1]
    w = float(eos.w)

    # validity, located; the theta-limiter
    rho_min_cell = j_e + int(np.argmin(d.rho[cells]))
    Gammabar2_min_face = j_e + int(np.argmin(d.Gammabar2[faces]))
    theta_binds = 0
    kernels = i.result.kernels
    if kernels is not None:
        theta_binds = int(np.sum(kernels.theta_scale[cells] < 1.0))

    # the Courant cell and the speeds at its faster face
    Lam_hat = np.maximum(sp.Lam[j_e:N], sp.Lam[j_e + 1 : N + 1])
    ratio = geo.dX[cells] / Lam_hat
    k = int(np.argmin(ratio))
    faster = j_e + k if sp.Lam[j_e + k] >= sp.Lam[j_e + k + 1] else j_e + k + 1

    # far zone
    far = X[j_e : N + 1] >= i.far_zone_from
    far_cells = far[:-1]  # a cell is far if its inner face is
    delta_rho = d.delta_rho[cells]
    delta_U = d.delta_U[faces] if j_e > 0 else np.concatenate(([0.0], d.delta_U[1:]))
    far_rho = float(np.max(np.abs(delta_rho[far_cells]))) if np.any(far_cells) else 0.0
    far_U = float(np.max(np.abs(delta_U[far]))) if np.any(far) else 0.0

    # outer boundary
    energy = boundary_energy(state, d, geo, bg, layout) if eos.is_radiation else float("nan")

    # centre and grid scale
    delta_U_1 = float(d.delta_U[max(j_e, 1)])
    inner = delta_rho[:4] if j_e == 0 else np.zeros(0)
    odd_even_inner = float(np.max(np.abs(alternating(inner)))) / rho_0 if inner.size >= 3 else 0.0

    # steepness
    log_rho = np.log(d.rho[cells])
    Xm = geo.Xm[cells]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_slope = np.abs(np.diff(log_rho) / np.diff(np.log(Xm)))
    steep = int(np.argmax(log_slope)) if log_slope.size else 0
    cell_jumps = np.abs(np.diff(log_rho))
    outside_3 = Xm[1:] > 3.0
    outside_5 = Xm[4:] > 5.0
    four_cell = np.exp(np.abs(log_rho[4:] - log_rho[:-4])) if N - j_e > 4 else np.zeros(0)
    supersonic = np.abs(sp.Theta[faces]) > sp.a[faces]

    # under-resolution
    half = np.flatnonzero(d.rho[cells] <= 0.5 * rho_0)
    core_cells = int(half[0]) if half.size else N - j_e
    core = slice(j_e, j_e + max(core_cells, 1))
    if kernels is not None:
        viscous = float(np.max(np.abs(kernels.q[core]) / (w * d.rho[core])))
        clipped = limiter_clipped(d.delta_rho, kernels.delta_rho_L, geo, layout)
        first = max(j_e, 1)  # face 0 carries no reconstruction
        jumps = kernels.delta_rho_R[first:N] - kernels.delta_rho_L[first:N]
        jump = float(np.max(np.abs(jumps) / d.rho_f[first:N]))
    else:
        viscous, jump = 0.0, 0.0
        clipped = np.zeros(N, dtype=bool)

    return Diagnostics(
        rho_min_cell=rho_min_cell,
        Gammabar2_min_face=Gammabar2_min_face,
        theta_binds=theta_binds,
        courant_ratio=float(ratio[k]),
        courant_cell=j_e + k,
        Theta_at_courant=float(sp.Theta[faster]),
        a_at_courant=float(sp.a[faster]),
        far_zone_delta_rho=far_rho,
        far_zone_delta_U=far_U,
        u_plus=u_plus,
        u_minus=u_minus,
        delta_m_N=float(d.delta_m[N]),
        delta_rho_N_1=float(d.delta_rho[N - 1]),
        boundary_energy=energy,
        delta_U_1=delta_U_1,
        odd_even_inner=odd_even_inner,
        grid_scale_cells=grid_scale_fraction(delta_rho),
        grid_scale_faces=grid_scale_fraction(delta_U[1:] if j_e == 0 else delta_U),
        steepest_log_slope=float(log_slope[steep]) if log_slope.size else 0.0,
        steepest_X=float(X[j_e + steep + 1]) if log_slope.size else 0.0,
        largest_cell_jump=float(np.max(cell_jumps)) if cell_jumps.size else 0.0,
        largest_cell_jump_outside_3=float(np.max(cell_jumps[outside_3])) if np.any(outside_3) else 0.0,
        largest_four_cell_ratio_outside_5=float(np.max(four_cell[outside_5])) if np.any(outside_5) else 1.0,
        supersonic_faces=int(np.sum(supersonic)),
        supersonic_outer_X=float(X[faces][supersonic].max()) if np.any(supersonic) else 0.0,
        core_cells=core_cells,
        viscous_over_pressure_core=viscous,
        clipped_cells=int(np.sum(clipped)),
        clipped_in_core=int(np.sum(clipped[core])),
        reconstruction_jump=jump,
    )


@dataclass(frozen=True)
class Diagnostics:
    """The second-tier columns of `MonitoredStep`, in its order; see there."""

    rho_min_cell: int
    Gammabar2_min_face: int
    theta_binds: int
    courant_ratio: float
    courant_cell: int
    Theta_at_courant: float
    a_at_courant: float
    far_zone_delta_rho: float
    far_zone_delta_U: float
    u_plus: float
    u_minus: float
    delta_m_N: float
    delta_rho_N_1: float
    boundary_energy: float
    delta_U_1: float
    odd_even_inner: float
    grid_scale_cells: float
    grid_scale_faces: float
    steepest_log_slope: float
    steepest_X: float
    largest_cell_jump: float
    largest_cell_jump_outside_3: float
    largest_four_cell_ratio_outside_5: float
    supersonic_faces: int
    supersonic_outer_X: float
    core_cells: int
    viscous_over_pressure_core: float
    clipped_cells: int
    clipped_in_core: int
    reconstruction_jump: float


NAN = float("nan")
UNSET = Diagnostics(
    -1,
    -1,
    -1,
    NAN,
    -1,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    NAN,
    -1,
    NAN,
    -1,
    NAN,
    -1,
    -1,
    NAN,
)
"""The second-tier columns on a step they are not evaluated: NaN, and `-1` for indices and counts."""
UNSET_COLUMNS = dataclasses.asdict(UNSET)  # converted once: a plain step must stay cheap


# --- the pieces ---


def alternating(f: FloatArray) -> FloatArray:
    """The alternating component `f_c - (f_{c-1} + f_{c+1}) / 2` of a field, on its interior points."""
    return f[1:-1] - 0.5 * (f[:-2] + f[2:])


def grid_scale_fraction(f: FloatArray) -> float:
    """The fraction of a field's root-mean-square deviation in its alternating component; `0` for a flat field."""
    total = float(np.sqrt(np.mean(f**2))) if f.size else 0.0
    if total == 0.0 or f.size < 3:
        return 0.0
    return float(np.sqrt(np.mean(alternating(f) ** 2))) / total


def limiter_clipped(
    delta_rho: FloatArray, delta_rho_L: FloatArray, geo: Geometry, layout: Layout
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Which retained cells the density limiter clipped: their reconstructed slope in `s` is not the unclipped one.

    The reconstruction's slope of cell `c` is `(rho_L,c+1 - rho_c) / (X_{c+1}^2 - sbar_c)`. An interior cell's unclipped
    slope is the mean of its two one-sided differences in `sbar`; the first and last retained cells' is their single
    adjacent difference. The theta-limiter scales any of them back near vacuum, and counts as a clip here. All are
    formed from the deviations `rho - 1`, as the reconstruction forms them, so that near FRW no rounding of the density
    to its FRW size reads as a clipped slope.
    """
    N, j_e = layout.N, layout.j_e
    clipped = np.zeros(N, dtype=bool)
    interior = slice(j_e + 1, N - 1)
    X2, sbar = geo.X**2, geo.sbar
    ds = X2[j_e + 2 : N] - sbar[interior]
    slope = (delta_rho_L[j_e + 2 : N] - delta_rho[interior]) / ds
    d_in = (delta_rho[interior] - delta_rho[j_e : N - 2]) / (sbar[interior] - sbar[j_e : N - 2])
    d_out = (delta_rho[j_e + 2 : N] - delta_rho[interior]) / (sbar[j_e + 2 : N] - sbar[interior])
    centred = 0.5 * (d_in + d_out)
    # A clipped slope differs from the centred one by a finite fraction of it; an unclipped one only by the round-off
    # of forming it from a face value, about 1e-16 |delta_rho| / ds.
    tolerance = 1e-12 * np.abs(centred) + 1e-13 * np.abs(delta_rho[interior]) / ds
    clipped[interior] = np.abs(slope - centred) > tolerance
    for e, nb in ((j_e, j_e + 1), (N - 1, N - 2)):  # the end cells, against their one-sided differences
        ds_e = X2[e + 1] - sbar[e]
        slope_e = (delta_rho_L[e + 1] - delta_rho[e]) / ds_e
        one_sided = (delta_rho[max(e, nb)] - delta_rho[min(e, nb)]) / (sbar[max(e, nb)] - sbar[min(e, nb)])
        clipped[e] = abs(slope_e - one_sided) > 1e-12 * abs(one_sided) + 1e-13 * abs(delta_rho[e]) / ds_e
    return clipped


def boundary_energy(state: State, d: Derived, geo: Geometry, bg: Background, layout: Layout) -> float:
    """The augmented boundary energy of Section 7.5 for radiation: the norm eq:num:norm with face `N`, plus `E_b`.

    `sum_c (9/4) c_s^2 dV_c delta_rho,c^2 + sum_{j=1}^{N} (X_j^3 dS_j / 2) (delta_U,j^2 + delta_m,j^2 / 8)
    + (1/4) c_s X_N^4 (W^2 + delta_m,N^2)`, on the unexcised grid; NaN once excised, where the norm is not printed.
    """
    from pbh.derived import Derived

    assert isinstance(d, Derived)
    N = layout.N
    if layout.excised:
        return float("nan")
    X = geo.X[: N + 1]
    delta_rho = d.delta_rho
    delta_U = d.delta_U[1:]
    delta_m = d.delta_m[1:]
    face_weight = 0.5 * X[1:] ** 3 * geo.dS[1 : N + 1]
    interior = float(np.sum(2.25 * bg.c_s**2 * geo.dV * delta_rho**2))
    faces = float(np.sum(face_weight * (delta_U**2 + delta_m**2 / 8.0)))
    E_b = 0.25 * bg.c_s * X[N] ** 4 * (state.W**2 + delta_m[-1] ** 2)
    return interior + faces + E_b
