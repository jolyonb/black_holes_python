//! One checked Runge-Kutta attempt in deviation form (`pbh/timestep.py`, `checked_step`; paper Section 7.6).
//!
//! The stages, the combinations between them and the checks on every stage input and on the result, with the same
//! operations in the same order as the Python. The retry policy that halves a refused step (`advance_checked`) stays in
//! the Python, which reads the refusal from here. The tableau comes from the Python (`ButcherTableau.floats`), and so
//! do the frames: one per stage after the first, at the stage times the Python forms, and one at the time the step
//! arrives at. A cell stored whole (`storage.rs`) advances by its whole rate, `DerivsResult::stored_rate`; the flags
//! are the Python's, the same for every stage of the attempt.

use crate::eos::{Background, EquationOfState};
use crate::equations::{DerivsResult, StageError, calc_derivs};
use crate::geometry::Geometry;
use crate::kernels::KernelSettings;
use crate::outer::OuterClosure;
use crate::state::FrwReference;
use crate::stencils::StencilWeights;
use crate::storage::{deviation_of, whole_of};

/// What a stage evaluates against at one time: the frame's geometry, weights and reference, its two scalar squares,
/// and the background at that time.
pub struct StageTime<'a> {
    pub geo: &'a Geometry,
    pub w: &'a StencilWeights,
    pub reference: &'a FrwReference,
    pub X_N_squared: f64,
    pub X_je_squared: f64,
    pub bg: Background,
}

/// The Scheme's settings, as a stage uses them.
pub struct Settings<'a> {
    pub eos: &'a EquationOfState,
    pub outer: &'a OuterClosure,
    pub kernels: &'a KernelSettings,
}

/// The stage at `y_FRW + delta y` from the packed deviation (`Scheme.evaluate_deviation`), the cells `whole` marks
/// holding their content itself.
pub fn evaluate_deviation(
    s: &Settings,
    t: &StageTime,
    dy: &[f64],
    whole: Option<&[bool]>,
) -> Result<DerivsResult, StageError> {
    // the length is the layout's: the caller checked it once for every vector the attempt forms
    let stored =
        t.w.layout
            .unpack(dy)
            .expect("a packed deviation of the layout's length");
    let state = whole_of(&stored, whole, &t.reference.state);
    let deviation = deviation_of(stored, whole, &t.reference.state.E);
    calc_derivs(
        &state,
        t.geo,
        &t.bg,
        s.eos,
        t.w,
        s.outer,
        s.kernels,
        &deviation,
        t.reference,
        t.X_N_squared,
        t.X_je_squared,
        whole,
    )
}

/// The scalars of a stage that the step's record reads (`StageFluxes.of`).
pub struct StageFluxes {
    pub F_N: f64,
    pub F_je: f64,
    pub M_total: f64,
    pub delta_F_N: f64,
    pub delta_M_total: f64,
}

impl StageFluxes {
    pub fn of(r: &DerivsResult, N: usize, j_e: usize) -> StageFluxes {
        StageFluxes {
            F_N: r.F[N],
            F_je: if j_e > 0 { r.F[j_e] } else { 0.0 },
            M_total: r.derived.M[N],
            delta_F_N: r.delta_F[N],
            delta_M_total: r.derived.delta_M[N],
        }
    }
}

/// One stage after the first, as the record keeps it (`timestep.Stage` less its time): the fluxes and the packed
/// rate of the stored numbers `k`.
pub struct Stage {
    pub fluxes: StageFluxes,
    pub k: Vec<f64>,
}

/// A refused attempt (`StepFailure`): the cause as `FailureCause`'s value, the stage (`2`, `3`, ... or `0` for the
/// result), the offending index (`-1` for a non-finite vector) and value.
pub struct Failure {
    pub cause: String,
    pub stage: usize,
    pub index: i64,
    pub value: f64,
}

/// What an attempt produced (`Attempt`): the stages it completed, and either the deviation arrived at with the rate
/// there, or the refusal.
pub struct Attempt {
    pub stages: Vec<Stage>,
    pub dy: Option<Vec<f64>>,
    pub result: Option<DerivsResult>,
    pub failure: Option<Failure>,
}

/// One checked attempt (`checked_step`) from the deviation `dy`, the first stage's packed rate `k1`, the tableau rows
/// `a` (the first empty) and weights `b`, the stage times after the first and the time arrived at, and the cells
/// stored whole, `whole`.
///
/// `Err` carries an outer closure's refusal, which the Python raises as `ValueError`.
#[allow(clippy::too_many_arguments)]
pub fn checked_step(
    s: &Settings,
    stage_times: &[StageTime],
    arrive: &StageTime,
    dy: &[f64],
    dxi: f64,
    k1: &[f64],
    a: &[Vec<f64>],
    b: &[f64],
    whole: Option<&[bool]>,
) -> Result<Attempt, String> {
    let evaluate = |t: &StageTime, dy_i: &[f64], stage: usize| -> Result<Result<DerivsResult, Failure>, String> {
        let place = if stage > 0 { "stage" } else { "result" };
        if !dy_i.iter().all(|v| v.is_finite()) {
            return Ok(Err(Failure {
                cause: format!("{place}_nonfinite"),
                stage,
                index: -1,
                value: f64::NAN,
            }));
        }
        match evaluate_deviation(s, t, dy_i, whole) {
            Ok(r) => Ok(Ok(r)),
            Err(StageError::NotHyperbolic(e)) => {
                let what = if e.value.is_finite() { e.field } else { "nonfinite" };
                Ok(Err(Failure {
                    cause: format!("{place}_{what}"),
                    stage,
                    index: e.index as i64,
                    value: e.value,
                }))
            }
            Err(StageError::Closure(message)) => Err(message),
        }
    };
    let layout = &arrive.w.layout;
    let (N, j_e) = (layout.N, layout.j_e);
    let mut k: Vec<Vec<f64>> = vec![k1.to_vec()];
    let mut stages = Vec::new();
    for (n, t) in stage_times.iter().enumerate() {
        let stage = n + 2; // the first stage is the accepted state's, which the caller holds
        let mut dy_i = dy.to_vec();
        for (a_ij, k_j) in a[n + 1].iter().zip(&k) {
            if *a_ij != 0.0 {
                let factor = dxi * a_ij;
                for e in 0..dy_i.len() {
                    dy_i[e] += factor * k_j[e];
                }
            }
        }
        match evaluate(t, &dy_i, stage)? {
            Ok(r) => {
                let fluxes = StageFluxes::of(&r, N, j_e);
                let k_i = t.w.layout.pack(r.stored_rate());
                stages.push(Stage { fluxes, k: k_i.clone() });
                k.push(k_i);
            }
            Err(failure) => {
                return Ok(Attempt {
                    stages,
                    dy: None,
                    result: None,
                    failure: Some(failure),
                });
            }
        }
    }
    // `dy + dxi * sum(b_i k_i)`, the sum from Python's integer zero, as `sum` forms it
    let mut total = vec![0.0; dy.len()];
    for (b_i, k_i) in b.iter().zip(&k) {
        for e in 0..total.len() {
            total[e] += b_i * k_i[e];
        }
    }
    let dy_new: Vec<f64> = dy.iter().zip(&total).map(|(y, s)| y + dxi * s).collect();
    match evaluate(arrive, &dy_new, 0)? {
        Ok(r) => Ok(Attempt {
            stages,
            dy: Some(dy_new),
            result: Some(r),
            failure: None,
        }),
        Err(failure) => Ok(Attempt {
            stages,
            dy: None,
            result: None,
            failure: Some(failure),
        }),
    }
}
