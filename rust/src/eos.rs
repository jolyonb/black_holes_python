//! The equation of state and the FRW background at one time, as the stage uses them (`pbh/eos.py`).
//!
//! Only the floats of `EquationOfState` cross into Rust, each formed once by the Python; the exact rationals stay
//! there. `Background` is the handful of scalars of `pbh.eos.Background` that a stage reads.

/// The constants of `EquationOfState` that a stage uses, as the Python formed them.
pub struct EquationOfState {
    /// `w` of `P = w rho`.
    pub w_float: f64,
    /// `alpha = 2 / (3 (1 + w))` (eq:asol).
    pub alpha_float: f64,
    /// `sqrt(w)`, the sound speed in units of light.
    pub sqrt_w: f64,
    /// `-w / (1 + w)`, the exponent of the algebraic lapse (eq:MSphinov); `-1/4` for radiation.
    pub lapse_exponent: f64,
    /// `2 - 3 alpha`, the source rate of the energy row (eq:num:energy).
    pub energy_source_rate: f64,
    /// Whether `w = 1/3`: the outgoing-wave closure's boundary ODE exists only then.
    pub is_radiation: bool,
}

/// The lapse of one density and its deviation from FRW (`EquationOfState.lapse_and_deviation`).
pub struct Lapse {
    /// `e^phi = rho ^ lapse_exponent`.
    pub ephi: f64,
    /// `e^phi - 1`, formed without subtracting one.
    pub delta_ephi: f64,
}

impl EquationOfState {
    /// The lapse `e^phi` and its deviation `e^phi - 1`, the latter without subtracting one
    /// (`EquationOfState.lapse_and_deviation`, one entry of it).
    ///
    /// For radiation, with `r = rho^(1/4)`, `e^phi - 1 = -delta_rho / (r (1 + r) (1 + r^2))`; for the stiff fluid,
    /// with `r = rho^(1/2)`, `-delta_rho / (r (1 + r))`; for any other `w`, `expm1(lapse_exponent log1p(delta_rho))`.
    /// The branches are chosen on the exact float, as in the Python.
    pub fn lapse_and_deviation(&self, rho: f64, delta_rho: f64) -> Lapse {
        if self.lapse_exponent == -0.25 {
            let r2 = rho.sqrt();
            let r = r2.sqrt();
            return Lapse {
                ephi: 1.0 / r,
                delta_ephi: -delta_rho / (r * (1.0 + r) * (1.0 + r2)),
            };
        }
        if self.lapse_exponent == -0.5 {
            let r = rho.sqrt();
            return Lapse {
                ephi: 1.0 / r,
                delta_ephi: -delta_rho / (r * (1.0 + r)),
            };
        }
        Lapse {
            ephi: rho.powf(self.lapse_exponent),
            delta_ephi: (self.lapse_exponent * delta_rho.ln_1p()).exp_m1(),
        }
    }
}

/// The background scalars of `pbh.eos.Background` that a stage reads, at the stage's time (no closure reads `xi`).
pub struct Background {
    /// The FRW value of `Gammabar^2`, `e^(2 (1 - alpha) xi)` (1 in flat spacetime).
    pub Gammabar2: f64,
    /// The background sound speed `c_s`.
    pub c_s: f64,
    /// The coefficient `h` of every Hubble, gravity and source term: 1 on FRW, 0 in flat spacetime.
    pub hubble: f64,
}
