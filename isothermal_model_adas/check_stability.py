"""Check SAVED full-ADAS roots at one n0 and lambda, without finding new branches.

Examples after installation, from the project root:
 python -B -m isothermal_model_adas.check_stability --scan full_adas_mol03_lambda --n0 1e14 --lambda-sh 7 --static-only
 python -B -m isothermal_model_adas.check_stability --scan full_adas_mol03_lambda --n0 1e14 --lambda-sh 7 --frequency-only
 python -B -m isothermal_model_adas.check_stability --scan full_adas_mol03_lambda --n0 1e14 --lambda-sh 7
 python -B -m isothermal_model_adas.check_stability --scan full_adas_mol03_lambda --n0 1e14 --lambda-sh 7 --s-real-max 2e5 --s-imag-max 2e5

By default each root uses 0<Re(s)<10*nu_ref, |Im(s)|<10*nu_ref.
Both limits can be overridden independently. This is a heuristic search box,
NEVER a proof of global stability. No changes to a scan or source configuration.
Spectral transfer uses intermediate QR by default; --transport direct restores
the old shooting implementation. QR does not alter the s=0 validation.
Contour sampling is adaptive by default; --contour-grid uniform restores
uniform doubling. Failed refinement remains unresolved, never a zero count.
"""

import argparse
import json
from math import isfinite
from pathlib import Path
import sys

from .linearized_model import LinearControls, LinearizedModel, saved_cases
from .spectrum import SpectralBox, find_modes
from .reference_frequency import reference_frequency


def select_box(real_min, real_max, imag_max, frequencies):
    """Explicit bounds win; otherwise use the per-profile frequency ceiling."""
    real_max = frequencies["s_real_max_per_s"] if real_max is None else real_max
    imag_max = real_max if imag_max is None else imag_max
    return SpectralBox(real_min,real_max,-imag_max,imag_max)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", type=Path, required=True, help="Existing scan directory (checkpoint and reports).")
    parser.add_argument("--n0", type=float, required=True, help="Saved neutral density, cm^-3.")
    parser.add_argument("--lambda-sh", type=float, required=True)
    parser.add_argument("--root-index", type=int, action="append", help="Saved index; omit to check ALL saved roots.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--static-only", action="store_true")
    mode.add_argument("--frequency-only", action="store_true",
                      help="Print nu_ref and planned bounds; no static sensitivities or spectrum.")
    parser.add_argument("--s-real-min", type=float, default=0.)
    parser.add_argument("--s-real-max", type=float)
    parser.add_argument("--s-imag-max", type=float, help="Symmetric imaginary bound in s^-1 (angular rate, NOT Hz).")
    parser.add_argument("--s-max-factor", type=float, default=10.,
                        help="Automatic real ceiling = factor*nu_ref (default 10). Ignored for explicit real ceiling.")
    parser.add_argument("--points", type=int, default=2049)
    parser.add_argument("--derivative-step", type=float, default=1e-5)
    parser.add_argument("--transport",choices=("qr","direct"),default="qr",
                        help="QR-stabilized transfer (default), or old direct shooting.")
    parser.add_argument("--qr-max-step",type=float,default=.025,
                        help="Maximum QR interval in x=sqrt((1-g)/(1-alpha)).")
    parser.add_argument("--qr-max-growth",type=float,default=2.,
                        help="Target maximum local |eigenvalue|*interval (default 2).")
    parser.add_argument("--qr-max-segments",type=int,default=20000,
                        help="Per-evaluation QR segment budget (default 20000).")
    parser.add_argument("--contour-grid",choices=("adaptive","uniform"),default="adaptive",
                        help="Adaptive contour refinement (default), or old uniform doubling.")
    parser.add_argument("--contour-max-points",type=int,default=8192,
                        help="Distinct s-point budget per adaptive contour (default 8192).")
    parser.add_argument("--contour-max-depth",type=int,default=20,
                        help="Maximum local bisection depth (default 20).")
    parser.add_argument("--contour-phase-step",type=float,default=.65,
                        help="Initial adaptive phase-step ceiling, radians (default .65).")
    parser.add_argument("--output", type=Path, help="New JSON report; existing files are not overwritten.")
    args = parser.parse_args(argv)
    if not all(isfinite(v) and v > 0 for v in (args.n0,args.lambda_sh,args.s_max_factor)):
        parser.error("n0, lambda and s-max-factor must be finite and positive.")
    if not isfinite(args.s_real_min):
        parser.error("s-real-min must be finite.")
    if args.s_real_max is not None and (not isfinite(args.s_real_max) or args.s_real_max <= args.s_real_min):
        parser.error("s-real-max must be finite and greater than s-real-min.")
    if args.s_imag_max is not None and (not isfinite(args.s_imag_max) or args.s_imag_max <= 0):
        parser.error("s-imag-max must be finite and positive.")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    if (args.contour_max_points < 16 or args.contour_max_depth < 1
            or not isfinite(args.contour_phase_step) or not 0 < args.contour_phase_step < 1.5707963267948966):
        parser.error("Invalid adaptive contour controls.")
    try:
        controls = LinearControls(points=args.points, derivative_step=args.derivative_step,
            qr_max_step=args.qr_max_step,qr_max_growth=args.qr_max_growth,
            qr_max_segments=args.qr_max_segments)
        roots = saved_cases(args.scan,args.n0,args.lambda_sh)
        if args.root_index is not None:
            roots = [bg for bg in roots if bg.root_index in args.root_index]
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    report = dict(n0_cm3=args.n0,lambda_sh=args.lambda_sh,transport=args.transport,
        contour_grid=args.contour_grid,
        assumptions=["fixed kinetic source and chi", "Ti=Te=T(t), uniform along the layer",
                     "prescribed neutrals and composition", "instantaneous ADAS/AMJUEL coefficients",
                     "moving source-free interface", "frozen stationary nondimensionalization"],
        static_columns=["dj_t/dT [1/eV]","da/dT [cm/eV]","dW_req/dT [kW/eV]"],
        mode_amplitudes=["delta log j_t","delta alpha","delta log theta"],
        roots=[])
    if not roots:
        report["status"] = "no_saved_verified_roots"
    for bg in roots:
        row = dict(case=bg.case_id,root_index=bg.root_index,T_eV=bg.seed["T_eV"],
                   a_cm=bg.alpha*bg.p["L_cm"],source=str(bg.report_path),
                   molecular_fraction=bg.p.get("molecular_fraction",0.))
        print(f"root {row['root_index']}: reconstructing T={row['T_eV']:.7g} eV",
              file=sys.stderr,flush=True)
        try:
            row["background"] = bg.verify()
            row["reference_frequency"] = reference_frequency(bg.p,bg.trial(),args.s_max_factor)
            box = select_box(args.s_real_min,args.s_real_max,args.s_imag_max,row["reference_frequency"])
            row["planned_box_s_per_s"] = vars(box)
            row["bounds_source"] = dict(
                real_max="manual" if args.s_real_max is not None else "factor*nu_ref",
                imag_max="manual" if args.s_imag_max is not None else "same_as_real_max")
            print(f"root {row['root_index']}: nu_ref={row['reference_frequency']['nu_ref_per_s']:.7g} s^-1; "
                  f"Re(s) in ({box.real_min:.7g},{box.real_max:.7g}), |Im(s)|<{box.imag_max:.7g}",
                  file=sys.stderr,flush=True)
            if args.frequency_only:
                row["status"] = "frequency_computed_spectrum_not_searched"
                report["roots"].append(row)
                continue
            model = LinearizedModel(bg,controls)
            row["static_check"] = model.check_static()
            if not row["static_check"]["passed"]:
                raise ValueError("s=0 validation failed; spectrum has not been searched.")
            if args.static_only:
                row["status"] = "static_check_passed_spectrum_not_searched"
            else:
                row["spectrum"] = find_modes(model,box,method=args.transport,
                    progress=lambda message: print(f"root {bg.root_index}: {message}",file=sys.stderr,flush=True),
                    contour_method=args.contour_grid,contour_options=dict(
                        max_points=args.contour_max_points,max_depth=args.contour_max_depth,
                        phase_limit=args.contour_phase_step))
                row["status"] = row["spectrum"]["status"]
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            row["status"] = "unresolved"
            row["error"] = str(exc)
            if getattr(exc,"diagnostic",None) is not None:
                row["error_details"] = exc.diagnostic
        report["roots"].append(row)
        print(f"root {row['root_index']}: T={row['T_eV']:.7g} eV: {row['status']}",
              file=sys.stderr,flush=True)
    encoded = json.dumps(report,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            stream.write(encoded+"\n")
    print(encoded)
    return int(not roots or any(r["status"] in ("unresolved","near_neutral_unresolved") for r in report["roots"]))


if __name__ == "__main__":
    raise SystemExit(main())
