"""Command line entry point for evolving black holes.

Constructs initial data (or loads it from a previous output file) and drives a Misner-Sharp evolution, writing
gnuplot-formatted output blocks to a data file.
"""

import argparse
from collections.abc import Sequence
from fractions import Fraction

from pbh.base import RADIATION_W, Status
from pbh.initial import compute_deltam0, growingmode, makegrid
from pbh.ms import MS, MSCommon, MSEulerian, MSLagrangian
from pbh.output import open_writer

HANDLERS: dict[str, type[MSCommon]] = {"eulerian": MSEulerian, "lagrangian": MSLagrangian}

#: Statuses in which an evolution is considered to have completed normally
SUCCESS_STATUSES = frozenset({Status.TIMEOUT, Status.BLACKHOLE_FORMED})


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="pbh", description="Evolve a primordial black hole using Misner-Sharp.")
    parser.add_argument(
        "initial_conditions",
        nargs="?",
        help="data file to load initial conditions from (first block); if omitted, construct the growing mode",
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=0,
        help="which snapshot of the initial conditions file to start from; negative counts from the end "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.dat",
        help="output file; a .npz suffix selects numpy archive output, anything else gnuplot text "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--scheme", choices=HANDLERS, default="eulerian", help="equations of motion to evolve (default: %(default)s)"
    )
    parser.add_argument("--viscosity", type=float, default=2.0, help="artificial viscosity (0 to disable)")
    parser.add_argument(
        "--viscosity-buffer",
        type=float,
        default=1.0,
        help="distance in r from the outer edge within which viscosity is switched off (default: %(default)s)",
    )
    parser.add_argument(
        "--viscosity-buffer-width",
        type=float,
        default=0.1,
        help="width in r of the viscosity switch-off (default: %(default)s)",
    )
    parser.add_argument("--max-time", type=float, default=7.0, help="xi at which to stop (default: %(default)s)")
    parser.add_argument("--output-step", type=float, default=0.1, help="xi between output blocks")
    parser.add_argument("--write-after", type=float, default=0.0, help="xi after which output is written")
    parser.add_argument("--no-black-hole-check", action="store_true", help="keep evolving after horizon formation")
    parser.add_argument("--enforce-timeout", action="store_true", help="stop once the longest mode peaks")
    parser.add_argument(
        "--w",
        type=Fraction,
        default=RADIATION_W,
        help="equation of state parameter P = w rho, as a rational such as 1/3; when restarting from a file it must "
        "match the value recorded there. Only w = 1/3 can currently be evolved from the command line "
        "(the exact outer boundary condition exists only for radiation); other values are accepted by the library API "
        "(default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress debugging output")

    grid = parser.add_argument_group("initial data (ignored when loading from file)")
    grid.add_argument("--gridpoints", type=int, default=600, help="number of gridpoints (default: %(default)s)")
    grid.add_argument("--squeeze", type=float, default=2.0, help="grid squeeze towards origin, 0 for uniform")
    grid.add_argument("--amax", type=float, default=10.0, help="outer edge of the grid (default: %(default)s)")
    grid.add_argument("--amplitude", type=float, default=0.175, help="Gaussian amplitude of deltam")
    grid.add_argument("--sigma", type=float, default=2.0, help="Gaussian width of deltam (default: %(default)s)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an evolution from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.w != RADIATION_W:
        # Fail before anything is built or written: MSCommon.udot_outer_boundary would raise the same complaint from
        # inside the integrator's first derivative evaluation.
        parser.error(
            f"--w: only w = 1/3 can be evolved from the command line (the exact outgoing-wave outer boundary "
            f"condition exists only for radiation), got {args.w}"
        )

    driver = MS(
        eomhandler=HANDLERS[args.scheme],
        black_hole_check=not args.no_black_hole_check,
        enforce_timeout=args.enforce_timeout,
        viscosity=args.viscosity or None,
        viscosity_buffer=args.viscosity_buffer,
        viscosity_buffer_width=args.viscosity_buffer_width,
        debug=not args.quiet,
        w=args.w,
    )

    if args.initial_conditions is not None:
        driver.load_initial_conditions(args.initial_conditions, snapshot=args.snapshot)
    else:
        grid = makegrid(gridpoints=args.gridpoints, squeeze=args.squeeze, Amax=args.amax)
        deltam0 = compute_deltam0(grid, amplitude=args.amplitude, sigma=args.sigma)
        r, u, m = growingmode(grid, deltam0, w=args.w)
        driver.set_initial_conditions(0.0, r, u, m)

    print("Evolver initialized. Beginning evolution!")
    with open_writer(args.output) as writer:
        driver.drive(output_step=args.output_step, writer=writer, max_time=args.max_time, write_after=args.write_after)
    print(f"Evolution complete! Status: {driver.status.name}")
    return 0 if driver.status in SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
