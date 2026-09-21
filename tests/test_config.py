"""Tests of pbh.config: reading, refusing, writing and building from the run configuration."""

import re
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from pbh.config import (
    ConfigError,
    EvolutionConfig,
    GridConfig,
    MapFamily,
    OuterChoice,
    OuterConfig,
    RunConfig,
    code_commit,
    load,
    save,
)
from pbh.kernels import DensityLimiter, Kernels, KernelSettings
from pbh.maps import IdentityMap, SinhStretch
from pbh.outer import HeldAtFrw, OutgoingWave, PenaltyStrengths
from pbh.stencils import FaceClosure
from pbh.timestep import Integrator, step_cap

EXAMPLE = Path(__file__).parent.parent / "examples" / "example.config.yaml"
EVOLUTION = "evolution: {xi_start: 0.0, xi_end: 1.0}\n"
MINIMAL = "grid: {N: 40, Rtilde_max: 4.0, scale: 2.0}\n" + EVOLUTION


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.config.yaml"
    path.write_text(text)
    return path


# --- reading ---


def test_the_example_file_is_the_documented_defaults_with_the_required_keys():
    config = load(EXAMPLE)
    minimal = RunConfig(
        grid=GridConfig(N=800, Rtilde_max=12.0, scale=3.0), evolution=EvolutionConfig(xi_start=0.0, xi_end=6.0)
    )
    assert config == minimal  # every other section of the example is its default


def test_a_minimal_file_takes_every_default(tmp_path: Path):
    config = load(write(tmp_path, MINIMAL))
    assert config.fluid.w == Fraction(1, 3)
    assert config.grid == GridConfig(N=40, Rtilde_max=4.0, scale=2.0)
    assert config.outer.closure is OuterChoice.OUTGOING_WAVE
    assert config.shocks.kernels is Kernels.PRODUCTION
    assert config.excision.face_closure is FaceClosure.FIRST_ORDER
    assert config.stepping.integrator is Integrator.RK4
    assert config.evolution == EvolutionConfig(xi_start=0.0, xi_end=1.0)


def test_every_key_can_be_set(tmp_path: Path):
    text = """
fluid: {w: 1/2}
grid: {map: uniform, N: 10, Rtilde_max: 3}
outer: {closure: held, tau_u: 1.0, tau_rho: 1.5, tau_W: 0.5}
shocks: {kernels: centred, density_limiter: minmod, c_v: 0.5, rho_floor: 1.0e-10}
excision: {face_closure: o2}
stepping: {integrator: ssprk3, courant_number: 0.4, cap_tolerance: 1.0e-6, cap_efolds: 3.0}
output: {snapshot_spacing: 0.1, snapshot_spacing_after: 0.02, flush_every: 50}
evolution: {xi_start: -1.0, xi_end: 2.5}
"""
    c = load(write(tmp_path, text))
    assert c.fluid.w == Fraction(1, 2)
    assert c.grid == GridConfig(N=10, Rtilde_max=3.0, map=MapFamily.UNIFORM)
    assert (c.outer.closure, c.outer.tau_u, c.outer.tau_rho, c.outer.tau_W) == (OuterChoice.HELD, 1.0, 1.5, 0.5)
    assert c.shocks.build() == KernelSettings(Kernels.CENTRED, DensityLimiter.MINMOD, 0.5, 1e-10)
    assert c.excision.face_closure is FaceClosure.SECOND_ORDER
    assert (c.stepping.integrator, c.stepping.courant_number) == (Integrator.SSPRK3, 0.4)
    assert (c.stepping.cap_tolerance, c.stepping.cap_efolds) == (1e-6, 3.0)
    assert (c.output.snapshot_spacing, c.output.snapshot_spacing_after, c.output.flush_every) == (0.1, 0.02, 50)
    assert c.evolution == EvolutionConfig(xi_start=-1.0, xi_end=2.5)


def test_an_integer_w_is_read_as_a_rational(tmp_path: Path):
    assert load(write(tmp_path, MINIMAL + "fluid: {w: 1}\n")).fluid.w == Fraction(1)


def test_a_saved_error_names_the_file_and_every_bad_key(tmp_path: Path):
    path = write(tmp_path, MINIMAL + "fluid: {w: x}\nshocks: {c_v: no}\n")
    with pytest.raises(ConfigError, match=rf"{path.name}.*\n.*fluid.w\n(.*\n)*shocks.c_v\n"):
        load(path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("colour: blue\n", "colour\n  Extra inputs are not permitted"),
        ("fluid: {w: 1/3, gamma: 2}\n", "fluid.gamma\n  Extra inputs are not permitted"),
        ("fluid: {w: one third}\n", "fluid.w\n  Input is not a valid fraction"),
        ("fluid: 3\n", "fluid\n  Input should be a valid dictionary"),
        ("stepping: {courant_number: true}\n", "stepping.courant_number\n  Input should be a valid number"),
        (
            "stepping: {courant_number: 1e-1}\n",
            "stepping.courant_number\n  Input should be a valid number",
        ),  # a string in YAML
        ("stepping: {courant_number: 1.5}\n", "stepping.courant_number\n  Input should be less than or equal to 1"),
        ("stepping: {integrator: euler}\n", "stepping.integrator\n  Input should be 'rk4' or 'ssprk3'"),
        ("shocks: {kernels: 3}\n", "shocks.kernels\n  Input should be 'production' or 'centred'"),
        ("output: {flush_every: 0}\n", "output.flush_every\n  Input should be greater than or equal to 1"),
    ],
)
def test_bad_keys_and_values_are_refused_by_name(tmp_path: Path, extra: str, message: str):
    with pytest.raises(ConfigError, match=message):
        load(write(tmp_path, MINIMAL + extra))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (EVOLUTION, "grid\n  Field required"),
        ("grid: {N: 40, Rtilde_max: 4.0, scale: 2.0}\n", "evolution\n  Field required"),
        ("grid: {Rtilde_max: 4.0, scale: 2.0}\n" + EVOLUTION, "grid.N\n  Field required"),
        (
            "grid: {N: 40.5, Rtilde_max: 4.0, scale: 2.0}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid.N\n  Input should be a valid integer",
        ),
        (
            "grid: {N: true, Rtilde_max: 4.0, scale: 2.0}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid.N\n  Input should be a valid integer",
        ),
        (
            "grid: {N: 1, Rtilde_max: 4.0, scale: 2.0}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid.N\n  Input should be greater than or equal to 2",
        ),
        (
            "grid: {N: 4, Rtilde_max: 0.0, scale: 2.0}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid.Rtilde_max\n  Input should be greater than 0",
        ),
        (
            "grid: {N: 4, Rtilde_max: 4.0}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid\n  Value error, scale is required for the sinh map",
        ),
        (
            "grid: {N: 4, Rtilde_max: 4.0, scale: -1.0}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid.scale\n  Input should be greater than 0",
        ),
        (
            "grid: {N: 4, Rtilde_max: 4.0, scale: 1.0, map: uniform}\nevolution: {xi_start: 0.0, xi_end: 1.0}\n",
            "grid\n  Value error, scale has no meaning for the uniform map",
        ),
        (
            "grid: {N: 4, Rtilde_max: 4.0, scale: 1.0}\nevolution: {xi_start: 1.0, xi_end: 1.0}\n",
            "evolution\n  Value error, xi_end .* must exceed",
        ),
        ("- a list\n", "must be a mapping of sections"),
    ],
)
def test_missing_and_inconsistent_keys_are_refused_by_name(tmp_path: Path, text: str, message: str):
    with pytest.raises(ConfigError, match=message):
        load(write(tmp_path, text))


# --- writing ---


def test_a_saved_configuration_is_complete_carries_its_provenance_and_reloads_unchanged(tmp_path: Path):
    config = load(write(tmp_path, MINIMAL))
    out = tmp_path / "saved.config.yaml"
    save(config, out)
    document = yaml.safe_load(out.read_text())
    sections = ["provenance", "fluid", "grid", "outer", "shocks", "excision", "stepping", "output", "evolution"]
    assert list(document) == sections
    assert document["grid"] == {"N": 40, "Rtilde_max": 4.0, "map": "sinh", "scale": 2.0}
    assert re.fullmatch(r"[0-9a-f]{12}(-dirty)?|unknown", document["provenance"]["code_commit"])
    assert document["provenance"]["written"].endswith("Z")
    assert document["fluid"] == {"w": "1/3"}
    assert document["shocks"] == {"kernels": "production", "density_limiter": "mc", "c_v": 1.0, "rho_floor": 1e-12}
    assert load(out) == config


def test_the_uniform_map_is_saved_without_a_scale(tmp_path: Path):
    config = RunConfig(
        grid=GridConfig(N=10, Rtilde_max=3.0, map=MapFamily.UNIFORM),
        evolution=EvolutionConfig(xi_start=0.0, xi_end=1.0),
    )
    out = tmp_path / "uniform.config.yaml"
    save(config, out)
    assert "scale" not in yaml.safe_load(out.read_text())["grid"]
    assert load(out) == config


def test_the_code_commit_is_unknown_outside_a_checkout(monkeypatch: pytest.MonkeyPatch):
    def no_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert code_commit() == "unknown"

    def not_a_repo(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal: not a git repository")

    monkeypatch.setattr(subprocess, "run", not_a_repo)
    assert code_commit() == "unknown"


# --- building ---


def test_the_sections_build_the_objects_they_describe():
    config = load(EXAMPLE)
    assert config.fluid.build().is_radiation
    assert config.grid.build() == SinhStretch(12.0, scale=3.0)
    assert config.outer.build() == OutgoingWave(PenaltyStrengths(2.0, 1.0, 0.0))
    assert config.shocks.build() == KernelSettings()
    assert config.stepping.cap(config.fluid.build()) == step_cap(config.fluid.build(), 1e-5, 4.0)
    held_uniform = RunConfig(
        grid=GridConfig(N=10, Rtilde_max=3.0, map=MapFamily.UNIFORM),
        evolution=EvolutionConfig(xi_start=0.0, xi_end=1.0),
        outer=OuterConfig(closure=OuterChoice.HELD),
    )
    assert held_uniform.grid.build() == IdentityMap(3.0)
    assert held_uniform.outer.build() == HeldAtFrw()


def test_the_scheme_is_assembled_from_the_sections():
    config = load(EXAMPLE)
    sch = config.scheme()
    assert sch.eos.is_radiation
    assert sch.map == SinhStretch(12.0, scale=3.0)
    assert sch.layout.N == 800
    assert sch.layout.j_e == 0
    assert sch.closure is FaceClosure.FIRST_ORDER
    assert isinstance(sch.outer, OutgoingWave)
    assert sch.settings == KernelSettings()
