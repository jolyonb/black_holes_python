//! The numpy element operations whose result is not fixed by IEEE arithmetic alone: the extrema and minmod.
//!
//! Everything else in the engine is `+ - * /`, `sqrt` and `abs`, which are correctly rounded in numpy and in Rust
//! alike, so the same expression in the same order gives the same value, with one exception on each side. The
//! lapse of a general `w` (neither radiation nor the stiff fluid; `EquationOfState::lapse_and_deviation`) calls
//! `powf`, `ln_1p` and `exp_m1`, which are not correctly rounded: Rust takes them from the platform's C library, and
//! numpy takes `power`, `log1p` and `expm1` from the same library on this machine (macOS on arm64), but may use its
//! own SIMD versions elsewhere (x86 Linux), so there the two engines can differ by an ulp or so in the lapse and what
//! follows from it (why `tests/test_rust_engine.py` allows a general `w` a lapse error of a few ulp, propagated
//! entry by entry). The other exception is the sign bit of a NaN: the compiler may fold a negation into a
//! neighbouring subtraction (`-(a - b) + c` as `c - (a - b)`), exact for every number, where numpy negates literally;
//! so a NaN the stage computes may carry the other sign (`pbh/rust_engine.py`).
//! The extrema are written out here because Rust's `f64::max` and `f64::min` do not propagate NaN, and numpy's do.
//!
//! The order of `+0` and `-0` on a tie is numpy's on arm64; numpy's x86 SIMD extrema order them differently, which is
//! why `tests/test_rust_engine.py` compares values by `==` and never the sign of a zero.

/// `np.maximum(a, b)` as numpy computes it on this machine: NaN if either is NaN (the first operand's if both are),
/// otherwise the larger, and `+0` over `-0` on a tie of zeros. Never `f64::max`, which returns the other operand
/// when one is NaN.
pub fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a > b {
        a
    } else if b > a {
        b
    } else if a.is_sign_positive() {
        a
    } else {
        b
    }
}

/// `np.minimum(a, b)`: NaN if either is NaN (the first operand's if both are), otherwise the smaller, and `-0` over
/// `+0` on a tie of zeros.
pub fn minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a < b {
        a
    } else if b < a {
        b
    } else if a.is_sign_negative() {
        a
    } else {
        b
    }
}

/// `kernels.minmod` of two slopes: the one of smallest modulus where both agree in sign, zero otherwise (and zero
/// where either is NaN, since then neither sign test holds).
pub fn minmod2(first: f64, second: f64) -> f64 {
    let smallest = minimum(first.abs(), second.abs());
    if first > 0.0 && second > 0.0 {
        smallest
    } else if first < 0.0 && second < 0.0 {
        -smallest
    } else {
        0.0
    }
}

/// `kernels.minmod` of three slopes, the minimum of the moduli taken in the order the Python takes it.
pub fn minmod3(first: f64, second: f64, third: f64) -> f64 {
    let smallest = minimum(minimum(first.abs(), second.abs()), third.abs());
    if first > 0.0 && second > 0.0 && third > 0.0 {
        smallest
    } else if first < 0.0 && second < 0.0 && third < 0.0 {
        -smallest
    } else {
        0.0
    }
}
