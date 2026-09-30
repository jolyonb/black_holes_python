"""When the outer boundary can have acted at a radius: the causal isolation of Section 8.6.

The outer face acts from the first step of a run, the penalty of eq:num:sat being live from the start, and whatever it
does travels inward. On the background, sound crosses one unit of `Rtilde` per unit of the sound horizon
`tau = c_s / (1 - alpha)` (eq:lin:tau) and light one per unit of `tau_L = tau / sqrt w`, the Hubble radius for
radiation (eq:numbh:causal, eq:numbh:causalL). A signal that leaves the outer face at `xi_0` has therefore not reached
the radius `r` by `xi` if

    Rtilde_max >= r + tau(xi) - tau(xi_0)            on sound, the estimate, which a shock outruns;
    Rtilde_max >= r + [tau(xi) - tau(xi_0)] / sqrt w  on light, the guarantee.

The run reports both, as the domain it would have needed and whether its own clears it, at formation and at the
read-out for the apparent horizon, and at its end for the origin. They are statements, not gates: the boundary
absorbs, so what returns after the sound cone arrives is small, and a production domain is not expected to clear the
light cone at the read-out.
"""

from typing import Any

from pbh.eos import Background, EquationOfState


def isolation(eos: EquationOfState, xi_since: float, xi: float, r: float, Rtilde_max: float) -> dict[str, Any]:
    """The domains that keep the radius `r` out of the boundary's reach until `xi`, on sound and on light.

    `xi_since` is when the outer face began to act, the start of the run or of the run it continues.
    """
    sound = Background.at(eos, xi).tau - Background.at(eos, xi_since).tau
    light = sound / eos.sqrt_w
    return {
        "r": r,
        "since": xi_since,
        "Rtilde_max": Rtilde_max,
        "needed_sound": r + sound,
        "needed_light": r + light,
        "isolated_sound": Rtilde_max >= r + sound,
        "isolated_light": Rtilde_max >= r + light,
    }
