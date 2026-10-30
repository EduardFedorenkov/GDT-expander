"""Compare an atomic lambda scan with the frozen local 36-pair reference.

Run from the project root:
    python -B tests/compare_local_reference.py atomic_lambda_test_1

Root matching uses (lambda_sh,n0) and minimum-cost matching in (T,a,j_t),
NOT local IDs or branch labels. No source data or scan settings are changed.
This regression comparison is not a proof of completeness of either scan.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_REFERENCE = Path(__file__).resolve().parent/'local_reference/atomic_lambda'
ROOT_FILE = 'roots_temperature_recombination.csv'
GRID_FILE = 'grid_temperature_recombination.csv'
ROOT_NUMERIC = ('T_eV', 'theta', 'a_cm', 'alpha', 'j_target',
    'net_particle_loss_fraction', 'recombination_count_per_inlet',
    'ionization_count_per_inlet', 'energy_ionization_fraction',
    'D', 'E_exc', 'R_rr', 'Q_rr', 'D_ion', 'E_ion')
# Absolute floors carry the units of each stored quantity (CGS and eV).
ABS_TOL = dict(T_eV=1e-8, a_cm=1e-6, theta=1e-11, alpha=1e-8,
    j_target=1e-8, net_particle_loss_fraction=1e-8,
    recombination_count_per_inlet=1e-10, ionization_count_per_inlet=1e-8,
    energy_ionization_fraction=1e-10)
RESIDUALS = ('R_J', 'R_M', 'R_E')
BALANCES = ('particle_balance_abs', 'momentum_balance_abs', 'energy_balance_abs')
SETTINGS = ('format_version', 'cx_mode', 'rr_scale', 'excitation_scale',
            'ionization_scale', 'inlet_momentum', 'geometry')
GRID_COUNTS = ('n_roots', 'n_trials', 'n_valid', 'candidate_cell_count', 'n_starts',
               'n_converged_starts', 'unresolved_numerical_count', 'failed_start_count')


def read_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f'Missing or duplicate CSV column names: {path}')
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'Malformed CSV row: {path}')
        return rows


def number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {field}: {row.get("case_id", "unknown case")}')
    return value


def key(row):
    return number(row, 'lambda_sh'), number(row, 'n0_cm3')


def load_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'))
    grid = read_table(folder/GRID_FILE)
    roots = read_table(folder/ROOT_FILE)
    pairs = {}
    for row in grid:
        pair = key(row)
        if pair in pairs:
            raise ValueError(f'Duplicate physical pair in {folder}: {pair}')
        pairs[pair] = row
    expected = {(float(lam), float(n0)) for lam in manifest['lambdas']
                for n0 in manifest['densities']}
    if set(pairs) != expected:
        raise ValueError(f'Incomplete/extra physical pairs in {folder}')
    groups = {pair: [] for pair in pairs}
    for row in roots:
        pair = key(row)
        if pair not in groups:
            raise ValueError(f'Root for an unknown physical pair: {pair}')
        groups[pair].append(row)
    for pair, row in pairs.items():
        count = number(row, 'n_roots')
        if count != len(groups[pair]):
            raise ValueError(f'CSV root count mismatch in {folder}: {pair}')
    return manifest, pairs, groups


def compare_runs(reference, candidate, *, rtol=1e-7):
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError('rtol must be finite and positive.')
    ref_manifest, ref_grid, ref_roots = load_run(reference)
    new_manifest, new_grid, new_roots = load_run(candidate)
    issues, diagnostic_changes = [], []
    for field in ('format_version', 'lambdas', 'densities', 'geometry',
                  'ionization_scale', 'parameters', 'scan_arguments'):
        if ref_manifest[field] != new_manifest[field]:
            issues.append(f'Manifest settings differ: {field}')
    # Code hashes must differ after a refactor; they are provenance, not physics.
    if set(ref_grid) != set(new_grid):
        issues.append('The sets of physical parameter pairs differ.')
    maxima = {name: 0. for name in ROOT_NUMERIC+RESIDUALS+BALANCES}
    matched = 0
    for pair in sorted(set(ref_grid) & set(new_grid)):
        a, b = ref_grid[pair], new_grid[pair]
        for name in ('cx_mode', 'rr_scale', 'format_version', 'excitation_scale',
                     'ionization_scale', 'n_roots', 'search_status'):
            if a[name] != b[name]:
                issues.append(f'{pair}: grid {name} differs ({a[name]} vs {b[name]})')
        for name in GRID_COUNTS[1:]:
            if number(a, name) != number(b, name):
                diagnostic_changes.append(f'{pair}: {name}: {a[name]} -> {b[name]}')
        for name in ('scan_statuses', 'attempt_statuses'):
            if json.loads(a[name]) != json.loads(b[name]):
                diagnostic_changes.append(f'{pair}: {name} differs')
        aa, bb = ref_roots[pair], new_roots[pair]
        if len(aa) != len(bb):
            issues.append(f'{pair}: root count differs ({len(aa)} vs {len(bb)})')
            continue
        if not aa:
            continue
        coordinates = ('T_eV', 'a_cm', 'j_target')
        x = np.array([[number(row, f) for f in coordinates] for row in aa])
        y = np.array([[number(row, f) for f in coordinates] for row in bb])
        scale = np.maximum(np.abs(x), [1e-8, 1e-6, 1e-8])
        cost = np.sum(((x[:, None, :]-y[None, :, :])/scale[:, None, :])**2, axis=2)
        rows, columns = linear_sum_assignment(cost)
        for i, j in zip(rows, columns):
            old, new = aa[i], bb[j]
            matched += 1
            for name in SETTINGS+('root_kind', 'outside_scan_j_range'):
                if old[name] != new[name]:
                    issues.append(f'{pair}, T={old["T_eV"]}: {name} differs')
            for name in ROOT_NUMERIC+RESIDUALS+BALANCES:
                u, v = number(old, name), number(new, name)
                delta = abs(u-v)
                maxima[name] = max(maxima[name], delta)
                tolerance = (1e-8 if name in RESIDUALS+BALANCES else
                             ABS_TOL.get(name, 1e-14)+rtol*abs(u))
                if delta > tolerance:
                    issues.append(f'{pair}, T={old["T_eV"]}: {name}, '
                                  f'abs difference {delta:.9g} > {tolerance:.9g}')
            for name in RESIDUALS:
                if abs(number(new, name)) > 1e-8:
                    issues.append(f'{pair}: inlet residual {name} exceeds 1e-8')
            for name in BALANCES:
                value = number(new, name)
                if not 0 <= value <= 5e-6:
                    issues.append(f'{pair}: balance {name} outside [0,5e-6]')
    return dict(passed=not issues, physical_pairs=len(ref_grid),
        reference_roots=sum(map(len, ref_roots.values())),
        candidate_roots=sum(map(len, new_roots.values())), matched_roots=matched,
        max_absolute_differences=maxima, issues=issues,
        diagnostic_changes=diagnostic_changes, rtol=rtol,
        reference=str(Path(reference).resolve()), candidate=str(Path(candidate).resolve()),
        reference_code_sha256=ref_manifest['code_sha256'],
        candidate_code_sha256=new_manifest['code_sha256'],
        note='Root counts and physical values are checked; IDs and file paths are ignored. '
             'Diagnostic counter changes are reported separately. '
             'No old per-case profiles are available for a profile comparison.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    try:
        result = compare_runs(args.reference, args.candidate, rtol=args.rtol)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n',
                                    encoding='utf-8')
    print('PASS' if result['passed'] else 'FAIL')
    print(f"Pairs: {result['physical_pairs']}; roots: {result['reference_roots']} -> "
          f"{result['candidate_roots']}; matched: {result['matched_roots']}")
    for name in ('T_eV', 'a_cm', 'j_target'):
        print(f"max abs difference {name}: {result['max_absolute_differences'][name]:.9g}")
    print('Diagnostic counter/status changes:', len(result['diagnostic_changes']))
    for issue in result['issues'][:20]:
        print(issue)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
