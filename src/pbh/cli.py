"""Command line entry point for evolving black holes.

Constructs initial data (or loads it from a previous output file) and drives a Misner-Sharp evolution, writing
gnuplot-formatted output blocks to a data file.
"""

import argparse
from collections.abc import Sequence

from pbh.initial import compute_deltam0, growingmode, makegrid
from pbh.ms import MS, MSCommon, MSEulerian, MSLagrangian

HANDLERS: dict[str, type[MSCommon]] = {"eulerian": MSEulerian, "lagrangian": MSLagrangian}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="pbh", description="Evolve a primordial black hole using Misner-Sharp.")
    parser.add_argument(
        "initial_conditions",
        nargs="?",
        help="data file to load initial conditions from (first block); if omitted, construct the growing mode",
    )
    parser.add_argument("-o", "--output", default="output.dat", help="output data file (default: %(default)s)")
    parser.add_argument(
        "--scheme", choices=HANDLERS, default="eulerian", help="equations of motion to evolve (default: %(default)s)"
    )
    parser.add_argument("--viscosity", type=float, default=2.0, help="artificial viscosity (0 to disable)")
    parser.add_argument("--max-time", type=float, default=7.0, help="xi at which to stop (default: %(default)s)")
    parser.add_argument("--output-step", type=float, default=0.1, help="xi between output blocks")
    parser.add_argument("--write-after", type=float, default=0.0, help="xi after which output is written")
    parser.add_argument("--no-black-hole-check", action="store_true", help="keep evolving after horizon formation")
    parser.add_argument("--enforce-timeout", action="store_true", help="stop once the longest mode peaks")
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
    args = build_parser().parse_args(argv)

    driver = MS(
        eomhandler=HANDLERS[args.scheme],
        black_hole_check=not args.no_black_hole_check,
        enforce_timeout=args.enforce_timeout,
        viscosity=args.viscosity or None,
        debug=not args.quiet,
    )

    if args.initial_conditions is not None:
        driver.load_initial_conditions(args.initial_conditions)
    else:
        grid = makegrid(gridpoints=args.gridpoints, squeeze=args.squeeze, Amax=args.amax)
        deltam0 = compute_deltam0(grid, amplitude=args.amplitude, sigma=args.sigma)
        r, u, m = growingmode(grid, deltam0)
        driver.set_initial_conditions(0.0, r, u, m)

    print("Evolver initialized. Beginning evolution!")
    with open(args.output, "w") as f:
        driver.drive(output_step=args.output_step, file_handle=f, max_time=args.max_time, write_after=args.write_after)
    print(f"Evolution complete! Status: {driver.status.name}")
    return 0 if driver.status.value >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
