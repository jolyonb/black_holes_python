"""The run configuration: one YAML file says everything about a run, and its sections build the objects that run it.

A run is three files, `name.config.yaml`, `name.initial.h5` and `name.evolution.h5`, and this module is the first of
them. The initial data are not configured here: they are the second file, which records the parameters that made them,
so that the configuration describes the scheme and the run and nothing about the perturbation. The configuration is a
tree of frozen pydantic models, one per section of the file, and each section knows how to build the runtime object it
describes: `FluidConfig.build()` is an `EquationOfState`, `GridConfig.build()` a `Map`, `OuterConfig.build()` an
`OuterClosure`, `ShockConfig.build()` a `KernelSettings`, and `RunConfig.scheme()` the `Scheme` of `timestep.py`
assembled from all of them. The driver reads the file and never sees a raw string.

    fluid:
      w: 1/3                    # the equation of state, P = w rho, as an exact rational
    grid:
      map: sinh                 # sinh (fine at the origin, coarse outside) or uniform
      N: 800                    # cells
      Rtilde_max: 12.0          # the outer face, in the scaled areal radius
      scale: 3.0                # the sinh stretch's scale; absent for the uniform map
    outer:
      closure: outgoing_wave    # outgoing_wave (Section 7.5) or held (U_N = X_N, a test closure)
      tau_u: 2.0                # the penalty strengths, Section 7.5
      tau_rho: 1.0
      tau_W: 0.0
    shocks:
      kernels: production       # production (Section 7.7) or centred (the base scheme, a test switch)
      density_limiter: mc       # mc or minmod
      c_v: 1.0
      rho_floor: 1.0e-12
      viscous_flux: density_weighted  # density_weighted (each side its own q/rho at its face density) or averaged
      cap_tension: true         # cap the viscous tension at the fluid pressure, q >= -w rho
    excision:
      face_closure: o1          # o1 (first order, production) or o2 (second order; unsafe near vacuum)
    stepping:
      integrator: rk4           # rk4 or ssprk3
      courant_number: 0.75
      cap_tolerance: 1.0e-5     # the step cap of eq:num:stepcap: relative error tolerance ...
      cap_efolds: 4.0           # ... over this many super-horizon e-folds
    output:
      snapshot_spacing: 0.02        # snapshots every this much in xi before formation ...
      snapshot_spacing_after: 0.02  # ... and every this much physical time, in Hubble times at formation, after
      flush_every: 200              # steps buffered before the step record is written
      monitor_every_step: false     # the full monitor record every step, not only at snapshots
    evolution:
      xi_end: 6.0                   # the run starts at the time of its initial data

Every key has the default shown except `N`, `Rtilde_max` and `xi_end`, which a run must state. A file
may omit any key with a default and may contain nothing else: an unknown key, a wrong type, or a value outside its
range is an error naming the key, never a warning. `save` writes the complete configuration with every default
filled in, under a `provenance` section giving the code's git commit and the time of writing; `load` accepts and
discards that section, so a saved configuration reruns as it was.

The sections are pydantic models in strict mode: a key gets the type it is declared with and nothing else, so `800`
is an int but `"800"` and `true` are not; an enumeration is given by its value; `w` is given as a string like `1/3`.
The range checks are validators on the section that owns them.

On YAML: floats need a digit on both sides of the point and after the exponent sign, `1.0e-5` and not `1e-5`,
which YAML reads as a string; the parser then reports the wrong type.
"""

import subprocess
from datetime import UTC, datetime
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from pbh.eos import RADIATION, EquationOfState, as_rational_w
from pbh.kernels import DensityLimiter, Kernels, KernelSettings, ViscousFlux
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, SinhStretch
from pbh.outer import HeldAtFrw, OuterClosure, OutgoingWave, PenaltyStrengths
from pbh.readout import ReadoutSettings
from pbh.stencils import FaceClosure
from pbh.timestep import COURANT_NUMBER, Integrator, Scheme, step_cap


class ConfigError(ValueError):
    """A configuration file that cannot be used, with the offending keys named."""


class Section(BaseModel):
    """What every section shares: frozen, unknown keys refused, and types taken strictly."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# --- the sections ---


class FluidConfig(Section):
    """The `fluid` section: the equation of state."""

    w: Fraction = Field(default=RADIATION, strict=False)
    """`P = w rho`, an exact rational given as a string like `1/3`; `1/3` is radiation."""

    def build(self) -> EquationOfState:
        """The equation of state."""
        return EquationOfState(as_rational_w(self.w))


class MapFamily(Enum):
    """The static base maps of `maps.py` a run can choose."""

    SINH = "sinh"
    """`SinhStretch`: fine at the origin, coarse in the background; the production map."""

    UNIFORM = "uniform"
    """`IdentityMap`: cells of equal width; a test convenience."""


class GridConfig(Section):
    """The `grid` section: the map, the number of cells and the outer radius (Section 7.1)."""

    N: int = Field(ge=2)
    """The number of cells."""

    Rtilde_max: float = Field(gt=0.0)
    """The scaled areal radius of the outer face."""

    map: MapFamily = Field(default=MapFamily.SINH, strict=False)
    """The map family."""

    scale: float | None = Field(default=None, gt=0.0)
    """The sinh stretch's scale, the radius below which the cells are nearly uniform; not given for `uniform`."""

    @model_validator(mode="after")
    def _scale_belongs_to_the_sinh_map(self) -> Self:
        if self.map is MapFamily.SINH and self.scale is None:
            raise ValueError("scale is required for the sinh map")
        if self.map is MapFamily.UNIFORM and self.scale is not None:
            raise ValueError("scale has no meaning for the uniform map")
        return self

    def build(self) -> Map:
        """The map."""
        if self.map is MapFamily.SINH:
            assert self.scale is not None  # the validator requires it
            return SinhStretch(self.Rtilde_max, scale=self.scale)
        return IdentityMap(self.Rtilde_max)


class OuterChoice(Enum):
    """The outer closures of `outer.py` a run can choose."""

    OUTGOING_WAVE = "outgoing_wave"
    """The exact outgoing-wave condition as a penalty (Section 7.5); the production closure."""

    HELD = "held"
    """The outer face held at FRW, `U_N = X_N`; a test closure that reflects."""


class OuterConfig(Section):
    """The `outer` section: the closure of the outer face and its penalty strengths (Section 7.5)."""

    closure: OuterChoice = Field(default=OuterChoice.OUTGOING_WAVE, strict=False)
    tau_u: float = 2.0
    tau_rho: float = 1.0
    tau_W: float = 0.0

    def build(self) -> OuterClosure:
        """The outer closure; the strengths are validated by `PenaltyStrengths`."""
        if self.closure is OuterChoice.HELD:
            return HeldAtFrw()
        return OutgoingWave(PenaltyStrengths(self.tau_u, self.tau_rho, self.tau_W))


class ShockConfig(Section):
    """The `shocks` section: the shock-capturing kernels and their constants (Section 7.7)."""

    kernels: Kernels = Field(default=Kernels.PRODUCTION, strict=False)
    density_limiter: DensityLimiter = Field(default=DensityLimiter.MC, strict=False)
    c_v: float = 1.0
    rho_floor: float = 1e-12
    viscous_flux: ViscousFlux = Field(default=ViscousFlux.DENSITY_WEIGHTED, strict=False)
    cap_tension: bool = True

    def build(self) -> KernelSettings:
        """The kernel settings."""
        return KernelSettings(
            self.kernels, self.density_limiter, self.c_v, self.rho_floor, self.viscous_flux, self.cap_tension
        )


class ExcisionConfig(Section):
    """The `excision` section: the post-formation map's switch-on and the excision-face closure (Sections 8.1, 8.3).

    The defaults are the recommended values of Table tab:numbh:params; the horizon finder and the excision rules
    join this section with their bites.
    """

    eta: float = Field(default=0.7, gt=0.0, lt=1.0)
    """The excision radius in units of the apparent-horizon radius at switch-on (admissible `0.7` to `0.8`, with
    `eta e^(-alpha tau_on)` at least about `0.6`)."""

    tau_on: float = Field(default=0.3, gt=0.0)
    """The ramp time of the switch-on, in `xi` (admissible `0.2` to `0.5`)."""

    c_t: float = Field(default=4.0, gt=0.0)
    """The transition's centre in units of the apparent-horizon label at switch-on, `x_t = c_t x_AH`."""

    c_Delta: float = Field(default=1.5, gt=0.0)
    """The transition's half-width in the same units, `Delta_t = c_Delta x_AH`; `x_t + Delta_t` must stay below `0.8`,
    which is checked at switch-on."""

    face_closure: FaceClosure = Field(default=FaceClosure.FIRST_ORDER, strict=False)
    """The excision-face closure: first order in production, the second-order rows as a switch."""

    enabled: bool = True
    """Whether to excise at all. Off, a collapse continues on its grid until the interior breaks the areal
    coordinate: the switch that lets an excised run be compared with the unexcised one from the same snapshot."""

    eta_r: float | None = Field(default=0.7, gt=0.0, lt=1.0)
    """The re-excision fraction: the face advances to `ceil(N eta_r x_AH)` as the horizon grows; `null` turns
    re-excision off (Table tab:numbh:params: optional, equal to `eta` when on)."""

    zone_extension_at: float = Field(default=0.8, gt=0.0, le=1.0)
    """A further zone is pinned when the horizon's label reaches this fraction of the outermost zone's inner edge,
    or jumps beyond it to a new trapped region."""


class ReadoutConfig(Section):
    """The `readout` section: reading the black-hole mass after formation (Section 8.5, `readout.py`).

    The first four are the paper's recipe; the target is the accuracy a run is carried to.
    """

    window: float = Field(default=0.3, gt=0.0)
    """The width in `xi` of the straight-line fit that gives the rate `omega`."""

    spacing: float = Field(default=0.005, gt=0.0)
    """The spacing in `xi` of the uniform resampling of `M_AH`."""

    floor: float = Field(default=2.0, gt=0.0)
    """The e-folds after formation before which no mass is read."""

    bar_span: float = Field(default=1.0, gt=0.0)
    """The e-folds over which the error bar takes the variation of `Q`."""

    target: float = Field(default=0.01, gt=0.0)
    """The error bar below which the mass is read."""

    stop: bool = True
    """Whether the run ends once the mass is read; `xi_end` stays the latest it may run to."""

    efficiency_tolerance: float = Field(default=0.2, gt=0.0)
    """The read-out is flagged when the measured efficiency differs from the Michel value by more than this fraction
    when the mass is read: steady accretion is then not established, and the reading deserves a look. The paper finds
    it within 6 per cent wherever `lambda_c eps <= 0.03`, overshooting to 1.17 on the way."""

    @model_validator(mode="after")
    def _the_settings_are_consistent(self) -> Self:
        self.build()
        return self

    def build(self) -> ReadoutSettings:
        """The read-out constants."""
        return ReadoutSettings(self.window, self.spacing, self.floor, self.bar_span, self.target)


class SteppingConfig(Section):
    """The `stepping` section: the integrator, the Courant number and the step cap (Section 7.6)."""

    integrator: Integrator = Field(default=Integrator.RK4, strict=False)
    courant_number: float = Field(default=COURANT_NUMBER, gt=0.0, le=1.0)
    cap_tolerance: float = 1e-5
    cap_efolds: float = 4.0

    def cap(self, eos: EquationOfState) -> float:
        """The step cap `Delta xi_max` of eq:num:stepcap for this equation of state."""
        return step_cap(eos, self.cap_tolerance, self.cap_efolds)


class OutputConfig(Section):
    """The `output` section: the snapshot schedule and the flush cadence of the evolution file (`output.py`)."""

    snapshot_spacing: float = Field(default=0.02, gt=0.0)
    """Before formation, the spacing of snapshots in `xi`; at N = 2000 a snapshot is 32 KB and costs a few tenths
    of a millisecond, so this is a few hundred snapshots and ten megabytes over a run."""

    snapshot_spacing_after: float = Field(default=0.02, gt=0.0)
    """After formation, the spacing of snapshots in physical time, in units of the Hubble time at formation."""

    flush_every: int = Field(default=200, ge=1)
    """How many steps the step record is buffered before it is written to disk."""

    monitor_every_step: bool = False
    """Whether the full monitor record of `monitors.py` is written every step rather than only at snapshots; the
    cheap first tier, the minima, the bookkeeping and the boundary scalars, is recorded every step regardless."""


class EvolutionConfig(Section):
    """The `evolution` section: when the run ends; it starts at the time its initial data carry."""

    xi_end: float


class RunConfig(Section):
    """The whole configuration: one field per section of the file, in the file's order."""

    fluid: FluidConfig = FluidConfig()
    grid: GridConfig
    outer: OuterConfig = OuterConfig()
    shocks: ShockConfig = ShockConfig()
    excision: ExcisionConfig = ExcisionConfig()
    readout: ReadoutConfig = ReadoutConfig()
    stepping: SteppingConfig = SteppingConfig()
    output: OutputConfig = OutputConfig()
    evolution: EvolutionConfig

    def scheme(self, map: Map | None = None, layout: Layout | None = None) -> Scheme:
        """The scheme this configuration describes; `map` replaces the base map and `layout` the unexcised layout."""
        return Scheme(
            self.fluid.build(),
            self.grid.build() if map is None else map,
            Layout(self.grid.N) if layout is None else layout,
            self.excision.face_closure,
            self.outer.build(),
            self.shocks.build(),
        )


# --- the file ---


def load(path: Path) -> RunConfig:
    """Read a configuration file, refusing unknown keys, wrong types and values out of range."""
    with path.open() as f:
        raw: object = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: the file must be a mapping of sections")
    document = {key: value for key, value in cast(dict[object, object], raw).items() if key != "provenance"}
    try:  # `provenance` is written by `save` and is not part of the configuration
        return RunConfig.model_validate(document)
    except ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e


class Provenance(Section):
    """The `provenance` section of a saved file: which code wrote it and when. Not part of the configuration."""

    code_commit: str
    """The git commit of the code, short form, `-dirty` appended if the tree had uncommitted changes, or `unknown`."""

    written: datetime
    """The time of writing, UTC."""

    @classmethod
    def now(cls) -> Self:
        """The provenance of a file written now by this code."""
        return cls(code_commit=code_commit(), written=datetime.now(UTC))


def save(config: RunConfig, path: Path) -> None:
    """Write the complete configuration, every default filled in, under its provenance."""
    document = {
        "provenance": Provenance.now().model_dump(mode="json"),
        **config.model_dump(mode="json", exclude_none=True),
    }
    with path.open("w") as f:
        yaml.safe_dump(document, f, sort_keys=False)


def code_commit() -> str:
    """The git commit of the code, short form, with `-dirty` appended if the tree has uncommitted changes.

    `unknown` if the code is not in a git checkout.
    """
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=here, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True)
    except OSError:
        return "unknown"
    if commit.returncode != 0 or status.returncode != 0:
        return "unknown"
    return commit.stdout.strip() + ("-dirty" if status.stdout.strip() else "")
