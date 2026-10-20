"""Compare the fixed generalized scan with the original committed scanner.

Read-only with respect to the project. The only output is an optional JSON
report. A checkout of the reference commit is NOT required.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

_parameters = None
_old_scan = None
_new_scan = None
_points = None
_b_points = None


def initialize(source_root, project, config, reference, points, b_points):
    global _parameters, _old_scan, _new_scan, _points, _b_points
    sys.path.insert(0, str(source_root))
    from simple_cx_model.config import load_cx_config
    from isothermal_cx_model.scan_neutral import find_density_solutions
    _parameters = load_cx_config(config)
    old_source = subprocess.check_output(
        ["git", "-c", "safe.directory="+str(Path(project)),
         "show", reference+":isothermal_cx_model/scan_neutral.py"],
        cwd=project, encoding="utf-8")
    module = types.ModuleType("isothermal_cx_model._initial_scan_reference")
    module.__package__ = "isothermal_cx_model"
    module.__file__ = str(Path(project)/"isothermal_cx_model/scan_neutral.py")
    exec(compile(old_source, module.__file__, "exec"), module.__dict__)
    _old_scan = module.find_density_solutions
    _new_scan = find_density_solutions
    _points, _b_points = points, b_points


def compare_case(case):
    lam, n0 = case
    parameters = dict(_parameters, lambda_sh=float(lam))
    started = time.perf_counter()
    old, _ = _old_scan(n0, parameters, n_points=_points,
                       b_points=_b_points, cx_mode="temperature")
    new, diagnostic = _new_scan(n0, parameters, n_points=_points,
                                b_points=_b_points, cx_mode="temperature")
    old = sorted(old, key=lambda row: row["T_eV"])
    new = sorted(new, key=lambda row: row["T_eV"])
    errors = []
    for left, right in zip(old, new):
        errors.append(dict(
            relative_T=abs(right["T_eV"]-left["T_eV"])/left["T_eV"],
            absolute_a_cm=abs(right["a_cm"]-left["a_cm"]),
            branch_equal=right["branch"] == left["branch"],
        ))
    return dict(lambda_sh=float(lam), n0_cm3=float(n0),
                old=old, new=new, count_equal=len(old)==len(new),
                errors=errors, statuses=diagnostic["statuses"],
                has_unresolved_points=diagnostic["has_unresolved_points"],
                seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root containing the fixed isothermal_cx_model.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", default="8c34d4e")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[5,6,7,8,9,10])
    parser.add_argument("--densities", type=float, nargs="+")
    parser.add_argument("--n0-count", type=int, default=31)
    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--b-points", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project, source_root = args.project.resolve(), args.source_root.resolve()
    config = (args.config or project/"simple_cx_model/cx_config.json").resolve()
    densities = args.densities or np.geomspace(1e12, 1e15, args.n0_count).tolist()
    cases = [(lam, n0) for lam in args.lambdas for n0 in densities]
    code_file = source_root/"isothermal_cx_model/scan_general.py"
    # This is an algorithm comparison with a common physical model. Refuse
    # to silently mix different coefficient or legacy-solver implementations.
    dependencies = [
        "isothermal_cx_model/cx.py", "isothermal_cx_model/solver.py",
        "simple_cx_model/config.py", "simple_cx_model/constants.py",
    ]
    for dependency in dependencies:
        historic = subprocess.check_output(
            ["git", "-c", "safe.directory="+str(project),
             "show", args.reference+":"+dependency],
            cwd=project, encoding="utf-8")
        current = (source_root/dependency).read_text(encoding="utf-8-sig")
        if historic.replace("\r\n", "\n").strip() != current.replace(
                "\r\n", "\n").strip():
            parser.error("Physical/reference dependency changed: "+dependency)
    metadata = dict(
        physical_config=json.loads(config.read_text(encoding="utf-8-sig")),
        reference_commit=subprocess.check_output(
            ["git", "-c", "safe.directory="+str(project),
             "rev-parse",args.reference],cwd=project,encoding="utf-8").strip(),
        fixed_scanner_sha256=hashlib.sha256(code_file.read_bytes()).hexdigest(),
        points=args.points, b_points=args.b_points, n_cases=len(cases),
        identical_reference_dependencies=dependencies,
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize,
        initargs=(source_root, project, config, args.reference,
                  args.points, args.b_points),
    ) as executor:
        pending = {executor.submit(compare_case, case): case for case in cases}
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(f"[{len(rows)}/{len(cases)}] lambda={row['lambda_sh']:g}, "
                  f"n0={row['n0_cm3']:.6e}: old={len(row['old'])}, "
                  f"new={len(row['new'])}, seconds={row['seconds']:.2f}",
                  flush=True)
    rows.sort(key=lambda row: (row["lambda_sh"], row["n0_cm3"]))
    comparisons = [error for row in rows for error in row["errors"]]
    summary = dict(
        cases=len(rows),
        mismatched_counts=sum(not row["count_equal"] for row in rows),
        total_old_roots=sum(len(row["old"]) for row in rows),
        total_new_roots=sum(len(row["new"]) for row in rows),
        max_relative_T=max((e["relative_T"] for e in comparisons), default=0),
        max_absolute_a_cm=max((e["absolute_a_cm"] for e in comparisons), default=0),
        branch_mismatches=sum(not e["branch_equal"] for e in comparisons),
        unresolved_cases=sum(row["has_unresolved_points"] for row in rows),
        seconds=time.perf_counter()-started,
    )
    report = dict(metadata=metadata, summary=summary, rows=rows)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["mismatched_counts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

