//! The Rust engine of `pbh`: one stage of the semi-discrete equations (paper Section 7.3), called from
//! `pbh.rust_engine`, which is the only Python that imports this module (`pbh_engine`).
//!
//! This file is the boundary with Python and nothing else: the settings come in once and are kept here, a frame is
//! built here from the map at the faces and its arrays handed back to the Python, and a stage takes the packed
//! deviation (or state) and the background scalars, and hands back every field the Python `DerivsResult` carries, as
//! fresh numpy arrays. The arithmetic lives in the modules below, one per Python module and
//! one function per Python function, so that the two can be read side by side. A stage outside the hyperbolic domain
//! raises the Python's own `pbh.derived.NotHyperbolicError`; a closure's refusal raises `ValueError` with the Python's
//! message, at the point where the Python raises it.

mod derived;
mod eos;
mod equations;
mod geometry;
mod horizon;
mod kernels;
mod layout;
mod numpy_like;
mod outer;
mod state;
mod stencils;
mod timestep;

use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use crate::eos::{Background, EquationOfState};
use crate::equations::{DerivsResult, StageError, calc_derivs};
use crate::geometry::Geometry;
use crate::kernels::{DensityLimiter, KernelSettings, Kernels, ViscousFlux};
use crate::layout::Layout;
use crate::numpy_like::c_pow;
use crate::outer::OuterClosure;
use crate::state::{FrwReference, State, deviation_from_frw, whole_state};
use crate::stencils::StencilWeights;

/// `pbh.derived.NotHyperbolicError`, looked up once.
static NOT_HYPERBOLIC_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// A numpy array's entries, copied (any strides).
fn to_vec(a: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    a.as_array().to_vec()
}

/// A fresh numpy array handed the vector without a copy.
fn to_numpy(py: Python<'_>, v: Vec<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, v).unbind()
}

/// What a stage needs at one time that does not depend on the state: the geometry, the stencil weights and the FRW
/// reference of one Python `Frame`, built here from the map at the faces, and the two scalar squares the Python
/// forms by `pow`.
///
/// Built once per frame (`Geometry.of`, `StencilWeights.of` and `FrwReference.of`, with the same operations in the
/// same order) and read back by the Python through the getters, each a fresh numpy array (or float), so that the
/// Python's `Frame` holds the same numbers the stages use. The constructor refuses, with `ValueError`, a layout the
/// Python's `Layout` would refuse and map values of mismatched lengths; after that no index a stage takes can fall
/// outside an array. What `Geometry.of` checks of the radii themselves (`X_0 = 0`, strictly increasing) the Python
/// checks before calling (`pbh.geometry.check_radii`).
#[pyclass(frozen, module = "pbh_engine")]
pub struct StageFrame {
    geo: Geometry,
    w: StencilWeights,
    reference: FrwReference,
    X_N_squared: f64,
    X_je_squared: f64,
}

#[pymethods]
impl StageFrame {
    /// The frame at one time from the map at the `N + 2` faces `0..N+1` (`Map.radii`), for the excision face `j_e`
    /// and the background coefficient `hubble` (`Background.hubble`).
    #[new]
    #[pyo3(signature = (settings, *, j_e, X, X_xi, hubble))]
    fn new(
        settings: &StageSettings,
        j_e: usize,
        X: PyReadonlyArray1<'_, f64>,
        X_xi: PyReadonlyArray1<'_, f64>,
        hubble: f64,
    ) -> PyResult<Self> {
        // The Python's Layout refuses anything else, and every loop below trusts it: check once, here, so that an
        // inconsistent frame is a ValueError and never an index out of bounds (which pyo3 raises as a PanicException,
        // a BaseException) or a wrapped subtraction.
        let faces_and_virtual = X.len();
        if faces_and_virtual < 4 || X_xi.len() != faces_and_virtual {
            return Err(PyValueError::new_err(format!(
                "the map values must be N + 2 >= 4 radii and as many velocities, got {} and {}",
                X.len(),
                X_xi.len()
            )));
        }
        let N = faces_and_virtual - 2;
        if j_e > N - 2 {
            return Err(PyValueError::new_err(format!(
                "a frame needs 0 <= j_e <= N - 2, got N = {N} and j_e = {j_e}"
            )));
        }
        let geo = Geometry::of(&to_vec(&X), &to_vec(&X_xi));
        let w = StencilWeights::of(&geo, Layout { N, j_e });
        let eos = &settings.eos;
        let reference = FrwReference::of(&geo, eos.alpha_float * eos.w_float * hubble, j_e, hubble);
        let X_N_squared = c_pow(geo.X[N], 2.0); // the closures' `X_N ** 2` on the float `X_N`
        let X_je_squared = c_pow(geo.X[j_e], 2.0); // `kernels.viscous_pressure`'s `X[j_e] ** 2` on the numpy scalar
        Ok(StageFrame {
            geo,
            w,
            reference,
            X_N_squared,
            X_je_squared,
        })
    }

    // --- the geometry (`Geometry`) ---

    #[getter]
    fn X(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.X.clone())
    }

    #[getter]
    fn X_xi(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.X_xi.clone())
    }

    #[getter]
    fn dV(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.dV.clone())
    }

    #[getter]
    fn dV_xi(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.dV_xi.clone())
    }

    #[getter]
    fn sbar(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.sbar.clone())
    }

    #[getter]
    fn dS(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.dS.clone())
    }

    #[getter]
    fn dX(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.dX.clone())
    }

    #[getter]
    fn Xm(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.Xm.clone())
    }

    #[getter]
    fn X2(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.X2.clone())
    }

    #[getter]
    fn X3(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.X3.clone())
    }

    #[getter]
    fn s_in(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.s_in.clone())
    }

    #[getter]
    fn s_out(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.geo.s_out.clone())
    }

    // --- the stencil weights (`StencilWeights`) ---

    #[getter]
    fn grad_s(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.w.grad_s.clone())
    }

    #[getter]
    fn centred_U(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.w.centred_U.clone())
    }

    #[getter]
    fn r_L(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.w.r_L.clone())
    }

    #[getter]
    fn r_R(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.w.r_R.clone())
    }

    #[getter]
    fn outer_U(&self) -> (f64, f64, f64) {
        let [a, b, c] = self.w.outer_U;
        (a, b, c)
    }

    #[getter]
    fn excision_U(&self) -> f64 {
        self.w.excision_U
    }

    #[getter]
    fn outer_rho(&self) -> (f64, f64, f64) {
        let [a, b, c] = self.w.outer_rho;
        (a, b, c)
    }

    // --- the FRW reference (`FrwReference`) ---

    #[getter]
    fn state_E(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.reference.state.E.clone())
    }

    #[getter]
    fn state_U(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.reference.state.U.clone())
    }

    #[getter]
    fn state_M_e(&self) -> f64 {
        self.reference.state.M_e
    }

    #[getter]
    fn rate_E(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.reference.rate.E.clone())
    }

    #[getter]
    fn rate_U(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.reference.rate.U.clone())
    }

    #[getter]
    fn rate_M_e(&self) -> f64 {
        self.reference.rate.M_e
    }

    #[getter]
    fn frw_speed(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.reference.frw_speed.clone())
    }

    #[getter]
    fn F_frw(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.reference.F_frw.clone())
    }
}

/// The Scheme's settings as a stage uses them: the equation of state's floats, the kernel switches and the closure.
#[pyclass(frozen, module = "pbh_engine")]
pub struct StageSettings {
    eos: EquationOfState,
    kernels: KernelSettings,
    outer: OuterClosure,
}

#[pymethods]
impl StageSettings {
    #[new]
    #[pyo3(signature = (
        *, w_float, alpha_float, sqrt_w, lapse_exponent, energy_source_rate, is_radiation,
        kernels, density_limiter, c_v, theta, viscous_flux, cap_tension,
        closure, tau_u=None, tau_rho=None, tau_W=None, rho_N=None, ephi_N=None,
    ))]
    fn new(
        w_float: f64,
        alpha_float: f64,
        sqrt_w: f64,
        lapse_exponent: f64,
        energy_source_rate: f64,
        is_radiation: bool,
        kernels: &str,
        density_limiter: &str,
        c_v: f64,
        theta: f64,
        viscous_flux: &str,
        cap_tension: bool,
        closure: &str,
        tau_u: Option<f64>,
        tau_rho: Option<f64>,
        tau_W: Option<f64>,
        rho_N: Option<f64>,
        ephi_N: Option<f64>,
    ) -> PyResult<Self> {
        let eos = EquationOfState {
            w_float,
            alpha_float,
            sqrt_w,
            lapse_exponent,
            energy_source_rate,
            is_radiation,
        };
        let kernels = KernelSettings {
            kernels: match kernels {
                "production" => Kernels::Production,
                "centred" => Kernels::Centred,
                _ => {
                    return Err(PyValueError::new_err(format!("unknown kernels {kernels:?}")));
                }
            },
            density_limiter: match density_limiter {
                "mc" => DensityLimiter::Mc,
                "minmod" => DensityLimiter::Minmod,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown density limiter {density_limiter:?}"
                    )));
                }
            },
            c_v,
            theta,
            viscous_flux: match viscous_flux {
                "averaged" => ViscousFlux::Averaged,
                "density_weighted" => ViscousFlux::DensityWeighted,
                _ => {
                    return Err(PyValueError::new_err(format!("unknown viscous flux {viscous_flux:?}")));
                }
            },
            cap_tension,
        };
        let outer = match (closure, tau_u, tau_rho, tau_W, rho_N, ephi_N) {
            ("held_at_frw", None, None, None, None, None) => OuterClosure::HeldAtFrw,
            ("outgoing_wave", Some(tau_u), Some(tau_rho), Some(tau_W), None, None) => {
                OuterClosure::OutgoingWave { tau_u, tau_rho, tau_W }
            }
            ("held_exterior", None, None, None, Some(rho_N), Some(ephi_N)) => {
                OuterClosure::HeldExterior { rho_N, ephi_N }
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "the closure {closure:?} and its parameters do not match"
                )));
            }
        };
        Ok(StageSettings { eos, kernels, outer })
    }
}

/// The derived fields of one stage (`pbh.derived.Derived`), as numpy arrays.
#[pyclass(frozen, module = "pbh_engine")]
pub struct DerivedOutput {
    #[pyo3(get)]
    rho: Py<PyArray1<f64>>,
    #[pyo3(get)]
    ephi: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_ephi: Py<PyArray1<f64>>,
    #[pyo3(get)]
    M: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_M: Py<PyArray1<f64>>,
    #[pyo3(get)]
    mt: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_rho: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_U: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_m: Py<PyArray1<f64>>,
    #[pyo3(get)]
    Gammabar2: Py<PyArray1<f64>>,
    #[pyo3(get)]
    rho_f: Py<PyArray1<f64>>,
    #[pyo3(get)]
    ephi_f: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_rho_f: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_ephi_f: Py<PyArray1<f64>>,
}

/// The speeds of one stage (`pbh.equations.Speeds`), as numpy arrays.
#[pyclass(frozen, module = "pbh_engine")]
pub struct SpeedsOutput {
    #[pyo3(get)]
    drift: Py<PyArray1<f64>>,
    #[pyo3(get)]
    Theta: Py<PyArray1<f64>>,
    #[pyo3(get)]
    cE: Py<PyArray1<f64>>,
    #[pyo3(get)]
    a: Py<PyArray1<f64>>,
    #[pyo3(get)]
    Lam: Py<PyArray1<f64>>,
}

/// What the kernels produced at one stage (`pbh.kernels.KernelResult`), as numpy arrays.
#[pyclass(frozen, module = "pbh_engine")]
pub struct KernelOutput {
    #[pyo3(get)]
    rho_L: Py<PyArray1<f64>>,
    #[pyo3(get)]
    rho_R: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_rho_L: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_rho_R: Py<PyArray1<f64>>,
    #[pyo3(get)]
    J: Py<PyArray1<f64>>,
    #[pyo3(get)]
    q: Py<PyArray1<f64>>,
    #[pyo3(get)]
    q_f: Py<PyArray1<f64>>,
    #[pyo3(get)]
    Q: Py<PyArray1<f64>>,
    #[pyo3(get)]
    F: Py<PyArray1<f64>>,
    #[pyo3(get)]
    theta_scale: Py<PyArray1<f64>>,
    #[pyo3(get)]
    Lam_plus: Py<PyArray1<f64>>,
    #[pyo3(get)]
    Lam_minus: Py<PyArray1<f64>>,
    #[pyo3(get)]
    v_L: Py<PyArray1<f64>>,
    #[pyo3(get)]
    v_R: Py<PyArray1<f64>>,
}

/// Everything one stage computed (`pbh.equations.DerivsResult`), as numpy arrays and floats.
#[pyclass(frozen, module = "pbh_engine")]
pub struct StageOutput {
    #[pyo3(get)]
    rate_E: Py<PyArray1<f64>>,
    #[pyo3(get)]
    rate_U: Py<PyArray1<f64>>,
    #[pyo3(get)]
    rate_W: f64,
    #[pyo3(get)]
    rate_M_e: f64,
    #[pyo3(get)]
    deviation_rate_E: Py<PyArray1<f64>>,
    #[pyo3(get)]
    deviation_rate_U: Py<PyArray1<f64>>,
    #[pyo3(get)]
    deviation_rate_W: f64,
    #[pyo3(get)]
    deviation_rate_M_e: f64,
    #[pyo3(get)]
    derived: Py<DerivedOutput>,
    #[pyo3(get)]
    speeds: Py<SpeedsOutput>,
    #[pyo3(get)]
    F: Py<PyArray1<f64>>,
    #[pyo3(get)]
    delta_F: Py<PyArray1<f64>>,
    #[pyo3(get)]
    kernels: Option<Py<KernelOutput>>,
}

impl StageOutput {
    /// Hand every field of a stage's result to numpy.
    fn from_result(py: Python<'_>, r: DerivsResult) -> PyResult<Self> {
        let d = r.derived;
        let derived = DerivedOutput {
            rho: to_numpy(py, d.rho),
            ephi: to_numpy(py, d.ephi),
            delta_ephi: to_numpy(py, d.delta_ephi),
            M: to_numpy(py, d.M),
            delta_M: to_numpy(py, d.delta_M),
            mt: to_numpy(py, d.mt),
            delta_rho: to_numpy(py, d.delta_rho),
            delta_U: to_numpy(py, d.delta_U),
            delta_m: to_numpy(py, d.delta_m),
            Gammabar2: to_numpy(py, d.Gammabar2),
            rho_f: to_numpy(py, d.rho_f),
            ephi_f: to_numpy(py, d.ephi_f),
            delta_rho_f: to_numpy(py, d.delta_rho_f),
            delta_ephi_f: to_numpy(py, d.delta_ephi_f),
        };
        let sp = r.speeds;
        let speeds = SpeedsOutput {
            drift: to_numpy(py, sp.drift),
            Theta: to_numpy(py, sp.Theta),
            cE: to_numpy(py, sp.cE),
            a: to_numpy(py, sp.a),
            Lam: to_numpy(py, sp.Lam),
        };
        let kernels = match r.kernels {
            None => None,
            Some(k) => {
                let output = KernelOutput {
                    rho_L: to_numpy(py, k.rho_L),
                    rho_R: to_numpy(py, k.rho_R),
                    delta_rho_L: to_numpy(py, k.delta_rho_L),
                    delta_rho_R: to_numpy(py, k.delta_rho_R),
                    J: to_numpy(py, k.J),
                    q: to_numpy(py, k.q),
                    q_f: to_numpy(py, k.q_f),
                    Q: to_numpy(py, k.Q),
                    F: to_numpy(py, k.F),
                    theta_scale: to_numpy(py, k.theta_scale),
                    Lam_plus: to_numpy(py, k.Lam_plus),
                    Lam_minus: to_numpy(py, k.Lam_minus),
                    v_L: to_numpy(py, k.v_L),
                    v_R: to_numpy(py, k.v_R),
                };
                Some(Py::new(py, output)?)
            }
        };
        Ok(StageOutput {
            rate_E: to_numpy(py, r.rate.E),
            rate_U: to_numpy(py, r.rate.U),
            rate_W: r.rate.W,
            rate_M_e: r.rate.M_e,
            deviation_rate_E: to_numpy(py, r.deviation_rate.E),
            deviation_rate_U: to_numpy(py, r.deviation_rate.U),
            deviation_rate_W: r.deviation_rate.W,
            deviation_rate_M_e: r.deviation_rate.M_e,
            derived: Py::new(py, derived)?,
            speeds: Py::new(py, speeds)?,
            F: to_numpy(py, r.F),
            delta_F: to_numpy(py, r.delta_F),
            kernels,
        })
    }
}

/// Run one stage on a whole state and its deviation, and turn a failure into the Python's exception.
fn run_stage(
    py: Python<'_>,
    frame: &StageFrame,
    settings: &StageSettings,
    bg: &Background,
    state: &State,
    deviation: &State,
) -> PyResult<StageOutput> {
    let result = calc_derivs(
        state,
        &frame.geo,
        bg,
        &settings.eos,
        &frame.w,
        &settings.outer,
        &settings.kernels,
        deviation,
        &frame.reference,
        frame.X_N_squared,
        frame.X_je_squared,
    );
    match result {
        Ok(r) => StageOutput::from_result(py, r),
        Err(StageError::NotHyperbolic(e)) => {
            let class = NOT_HYPERBOLIC_ERROR.import(py, "pbh.derived", "NotHyperbolicError")?;
            Err(PyErr::from_value(class.call1((e.field, e.index, e.value))?))
        }
        Err(StageError::Closure(message)) => Err(PyValueError::new_err(message)),
    }
}

/// One stage at the state `y_FRW + delta y` (`Scheme.evaluate_deviation`), from the packed deviation `dy`.
#[pyfunction]
#[pyo3(signature = (frame, settings, Gammabar2, c_s, hubble, dy))]
fn stage_deviation(
    py: Python<'_>,
    frame: &StageFrame,
    settings: &StageSettings,
    Gammabar2: f64,
    c_s: f64,
    hubble: f64,
    dy: PyReadonlyArray1<'_, f64>,
) -> PyResult<StageOutput> {
    let bg = Background { Gammabar2, c_s, hubble };
    let deviation = frame.w.layout.unpack(&to_vec(&dy)).map_err(PyValueError::new_err)?;
    let state = whole_state(&frame.reference, &deviation);
    run_stage(py, frame, settings, &bg, &state, &deviation)
}

/// One stage at the packed whole state `y` (`Scheme.evaluate`), its deviation recovered as `deviation_from_frw` does.
#[pyfunction]
#[pyo3(signature = (frame, settings, Gammabar2, c_s, hubble, y))]
fn stage_state(
    py: Python<'_>,
    frame: &StageFrame,
    settings: &StageSettings,
    Gammabar2: f64,
    c_s: f64,
    hubble: f64,
    y: PyReadonlyArray1<'_, f64>,
) -> PyResult<StageOutput> {
    let bg = Background { Gammabar2, c_s, hubble };
    let state = frame.w.layout.unpack(&to_vec(&y)).map_err(PyValueError::new_err)?;
    let deviation = deviation_from_frw(&state, &frame.reference);
    run_stage(py, frame, settings, &bg, &state, &deviation)
}

/// A completed stage as the Python reads it: `(F_N, F_je, M_total, delta_F_N, delta_M_total, k)`.
type StageRecord = (f64, f64, f64, f64, f64, Py<PyArray1<f64>>);

/// One checked attempt's outcome (`pbh.timestep.Attempt`), read by the Python through the getters.
#[pyclass(frozen, module = "pbh_engine")]
pub struct AttemptOutput {
    stages: Vec<timestep::Stage>,
    dy: Option<Vec<f64>>,
    #[pyo3(get)]
    result: Option<Py<StageOutput>>,
    failure: Option<timestep::Failure>,
}

#[pymethods]
impl AttemptOutput {
    /// Every stage after the first that the attempt completed: `(F_N, F_je, M_total, delta_F_N, delta_M_total, k)`.
    #[getter]
    fn stages(&self, py: Python<'_>) -> Vec<StageRecord> {
        self.stages
            .iter()
            .map(|s| {
                let f = &s.fluxes;
                (
                    f.F_N,
                    f.F_je,
                    f.M_total,
                    f.delta_F_N,
                    f.delta_M_total,
                    to_numpy(py, s.k.clone()),
                )
            })
            .collect()
    }

    /// The deviation arrived at, or `None` if the attempt was refused.
    #[getter]
    fn dy(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.dy.as_ref().map(|dy| to_numpy(py, dy.clone()))
    }

    /// The refusal `(cause, stage, index, value)`, or `None`.
    #[getter]
    fn failure(&self) -> Option<(String, usize, i64, f64)> {
        self.failure
            .as_ref()
            .map(|f| (f.cause.clone(), f.stage, f.index, f.value))
    }
}

/// A frame and its background as a stage evaluates against them.
fn stage_time<'a>(frame: &'a StageFrame, bg: (f64, f64, f64)) -> timestep::StageTime<'a> {
    timestep::StageTime {
        geo: &frame.geo,
        w: &frame.w,
        reference: &frame.reference,
        X_N_squared: frame.X_N_squared,
        X_je_squared: frame.X_je_squared,
        bg: Background {
            Gammabar2: bg.0,
            c_s: bg.1,
            hubble: bg.2,
        },
    }
}

/// One checked Runge-Kutta attempt in deviation form (`timestep.checked_step`): the stages after the first on
/// `frames` with their backgrounds `(Gammabar2, c_s, hubble)`, and the result on `arrive`, from the deviation `dy`
/// and the first stage's packed rate `k1`, with the tableau's rows `a` and weights `b` as floats.
#[pyfunction]
#[pyo3(signature = (settings, frames, backgrounds, arrive, arrive_background, dy, dxi, k1, a, b))]
#[allow(clippy::too_many_arguments)]
fn checked_step(
    py: Python<'_>,
    settings: &StageSettings,
    frames: Vec<PyRef<'_, StageFrame>>,
    backgrounds: Vec<(f64, f64, f64)>,
    arrive: &StageFrame,
    arrive_background: (f64, f64, f64),
    dy: PyReadonlyArray1<'_, f64>,
    dxi: f64,
    k1: PyReadonlyArray1<'_, f64>,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
) -> PyResult<AttemptOutput> {
    let layout = &arrive.w.layout;
    let size = layout.size();
    let same_layout = frames
        .iter()
        .all(|f| f.w.layout.N == layout.N && f.w.layout.j_e == layout.j_e);
    let stages = frames.len() + 1;
    if !same_layout
        || backgrounds.len() != frames.len()
        || a.len() != stages
        || b.len() != stages
        || a.iter().enumerate().any(|(n, row)| row.len() != n)
    {
        return Err(PyValueError::new_err(
            "an attempt needs one frame and background per stage after the first, all on one layout, and a tableau \
             of as many stages",
        ));
    }
    if dy.len() != size || k1.len() != size {
        return Err(PyValueError::new_err(format!(
            "expected packed vectors of length {size}, got {} and {}",
            dy.len(),
            k1.len()
        )));
    }
    let s = timestep::Settings {
        eos: &settings.eos,
        outer: &settings.outer,
        kernels: &settings.kernels,
    };
    let times: Vec<timestep::StageTime> = frames
        .iter()
        .zip(&backgrounds)
        .map(|(f, bg)| stage_time(f, *bg))
        .collect();
    let attempt = timestep::checked_step(
        &s,
        &times,
        &stage_time(arrive, arrive_background),
        &to_vec(&dy),
        dxi,
        &to_vec(&k1),
        &a,
        &b,
    )
    .map_err(PyValueError::new_err)?;
    let result = match attempt.result {
        Some(r) => Some(Py::new(py, StageOutput::from_result(py, r)?)?),
        None => None,
    };
    Ok(AttemptOutput {
        stages: attempt.stages,
        dy: attempt.dy,
        result,
        failure: attempt.failure,
    })
}

/// The horizon finder's numbers on one slice (`pbh.horizon.Trapping`), read by the Python through the getters.
#[pyclass(frozen, module = "pbh_engine")]
pub struct TrappingOutput {
    trapping: horizon::Trapping,
}

#[pymethods]
impl TrappingOutput {
    #[getter]
    fn h(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        to_numpy(py, self.trapping.h.clone())
    }

    #[getter]
    fn trapped_faces(&self) -> usize {
        self.trapping.trapped_faces
    }

    /// Every sign change `(j, t, outer)`, from the origin outward.
    #[getter]
    fn crossings(&self) -> Vec<(usize, f64, bool)> {
        self.trapping.crossings.iter().map(|c| (c.j, c.t, c.outer)).collect()
    }

    #[getter]
    fn margin(&self) -> f64 {
        self.trapping.margin
    }

    #[getter]
    fn margin_face(&self) -> usize {
        self.trapping.margin_face
    }

    #[getter]
    fn core_margin(&self) -> f64 {
        self.trapping.core_margin
    }

    #[getter]
    fn core_margin_face(&self) -> usize {
        self.trapping.core_margin_face
    }

    #[getter]
    fn outer_face_trapped(&self) -> bool {
        self.trapping.outer_face_trapped
    }
}

/// The trapping function on the retained faces `j_e..N` and what the finder reads from it (`horizon.trapping`), from
/// the face velocities and `Gammabar^2`, both of length `N + 1`.
#[pyfunction]
#[pyo3(signature = (U, Gammabar2, j_e))]
fn trapping(
    U: PyReadonlyArray1<'_, f64>,
    Gammabar2: PyReadonlyArray1<'_, f64>,
    j_e: usize,
) -> PyResult<TrappingOutput> {
    let faces = U.len();
    if Gammabar2.len() != faces || faces < 3 || j_e > faces - 3 {
        return Err(PyValueError::new_err(format!(
            "the finder needs U and Gammabar2 of one length N + 1 >= 3 and 0 <= j_e <= N - 2, got {} and {} with \
             j_e = {j_e}",
            faces,
            Gammabar2.len()
        )));
    }
    let result = horizon::trapping(&to_vec(&U), &to_vec(&Gammabar2), j_e).map_err(PyValueError::new_err)?;
    Ok(TrappingOutput { trapping: result })
}

/// The module `pbh_engine`.
#[pymodule]
fn pbh_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StageFrame>()?;
    m.add_class::<StageSettings>()?;
    m.add_class::<StageOutput>()?;
    m.add_class::<DerivedOutput>()?;
    m.add_class::<SpeedsOutput>()?;
    m.add_class::<KernelOutput>()?;
    m.add_function(wrap_pyfunction!(stage_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(stage_state, m)?)?;
    m.add_class::<TrappingOutput>()?;
    m.add_class::<AttemptOutput>()?;
    m.add_function(wrap_pyfunction!(checked_step, m)?)?;
    m.add_function(wrap_pyfunction!(trapping, m)?)?;
    Ok(())
}
