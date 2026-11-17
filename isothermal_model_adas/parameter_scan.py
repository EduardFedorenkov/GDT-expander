"""Batch the existing ADAS scanner; no new plasma equations or root solver.

One fresh 3D survey per (lambda_sh,n0), separate child process and output.
Only checked, fully completed cases enter aggregate tables/checkpoints.
"""

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import scan_stationary as single
from .atomic_coefficients import ATOMIC_MODELS, code_hashes
from .config import neutral_densities
from .profiles import MOLECULAR_PROFILE_FIELDS
from .recombination_adas import IONIZATION_ENERGY_ERG


SCAN_KEYS = ('atomic_model', 'rr_scale', 'ionization_scale', 'excitation_scale', 'inlet_momentum',
    'temperature_min', 'temperature_max', 'temperature_count', 'j_min', 'j_max',
    'j_count', 'j_grid', 'alpha_min', 'alpha_max', 'alpha_count', 'max_starts',
    'separation', 'guess', 'iterations', 'points', 'map_points', 'residual_tol',
    'balance_tol', 'dedup_tol', 'rtol', 'atol', 'max_step', 'j_axis', 'alpha_slice', 'no_plot')
CHOICES = dict(atomic_model=ATOMIC_MODELS, inlet_momentum=('auto', 'kinetic', 'constant'),
               j_grid=('log', 'linear'), j_axis=('log', 'linear', 'loss'))
ROOT_FIELDS = ('case_id', 'solution_id', 'local_solution_id', 'lambda_sh', 'n0_cm3',
    'n_d_cm3', 'n_d2_cm3',
    'T_eV', 'theta', 'a_cm', 'alpha', 'j_target', 'net_particle_change', 'D',
    'atomic_model', 'rr_scale', 'ionization_scale', 'excitation_scale', 'inlet_momentum',
    'recombination_count_per_inlet', 'ionization_count_per_inlet',
    'energy_loss_fraction', 'cx_energy_loss_fraction', 'excitation_energy_loss_fraction',
    'ionization_energy_loss_fraction', 'effective_energy_loss_fraction',
    'mar_count_per_inlet', 'mad_count_per_inlet', 'mai_count_per_inlet',
    'mar_energy_loss_fraction', 'mad_energy_loss_fraction', 'mai_energy_loss_fraction',
    'R_J', 'R_M', 'R_E', 'particle_balance_abs', 'momentum_balance_abs', 'energy_balance_abs',
    'outside_scan_j_range', 'outside_scan_alpha_range', 'case_report', 'profile_file')
GRID_FIELDS = ('case_id', 'lambda_sh', 'n0_cm3', 'n_d_cm3', 'n_d2_cm3', 'atomic_model', 'n_roots', 'search_status',
    'n_trials', 'n_valid', 'n_starts', 'n_converged_starts', 'failed_start_count',
    'best_sample_norm', 'unresolved_numerical_count', 'attempt_statuses', 'case_report')


def scan_defaults():
    defaults = vars(single.build_parser().parse_args([]))
    result = {key: defaults[key] for key in SCAN_KEYS}
    result['temperature_max'] = 150.  # Current physical search convention.
    return result


def parameter_grid(values, name, *, allow_zero=False):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or np.any(values < 0) or (not allow_zero and np.any(values == 0))):
        raise ValueError(f'{name}: need a finite nonempty positive grid'
                         + (' (zero is also allowed).' if allow_zero else '.'))
    return np.unique(values)


def _options(overrides):
    result = scan_defaults()
    overrides = {} if overrides is None else dict(overrides)
    if set(overrides)-set(result):
        raise ValueError(f'Unknown or batch-controlled scan options: {sorted(set(overrides)-set(result))}')
    result.update(overrides)
    for key, default in scan_defaults().items():
        value = result[key]
        if key in ('guess', 'alpha_slice'):
            continue
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be boolean.')
        elif isinstance(default, int):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f'{key} must be an integer.')
            result[key] = int(value)
        elif isinstance(default, float):
            result[key] = float(value)
            if not np.isfinite(result[key]):
                raise ValueError(f'{key} must be finite.')
    for key, choices in CHOICES.items():
        if result[key] not in choices:
            raise ValueError(f'Invalid {key}.')
    if result['alpha_slice'] is not None:
        value = result['alpha_slice']
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError('alpha_slice must be an integer or None.')
        result['alpha_slice'] = int(value)
    guesses = np.asarray(result['guess'], dtype=float)
    if guesses.size == 0:
        guesses = guesses.reshape(0, 3)
    if guesses.ndim != 2 or guesses.shape[1] != 3 or not np.isfinite(guesses).all():
        raise ValueError('guess must contain finite (T_eV,j_t,alpha) triples.')
    result['guess'] = guesses.tolist()
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _inside(directory, relative):
    path = (directory/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f'Artifact path escapes its output directory: {relative}')
    return path


def _run_scan(command, cwd, log_path):
    """No shell or visible helper window. Relay log; clean up on normal Ctrl+C."""
    with Path(log_path).open('x', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=cwd, env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def _option_arguments(options):
    arguments = []
    for key, value in options.items():
        flag = '--'+key.replace('_', '-')
        if key == 'guess':
            for triple in value:
                arguments.extend([flag, *map(str, triple)])
        elif isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif value is not None:
            arguments.extend([flag, str(value)])
    return arguments


def _active_hashes(signature, density):
    hashes = signature['data_sha256']
    return {key: value for key, value in hashes.items()
            if density > 0 or key in ('acd12_h.dat', 'prb12_h.dat')}


def _read_case(path, lam, density, signature, options):
    """Do not confuse a failed/partial process or old Case-A report with zero roots."""
    report = _json(path)
    p = dict(signature['parameters'], lambda_sh=float(lam), n0_cm3=float(density))
    gas = neutral_densities(p)
    model_options = {key: options[key] for key in ('atomic_model', 'rr_scale', 'ionization_scale', 'excitation_scale')}
    model_options['inlet_momentum'] = signature['inlet_momentum']
    if (report.get('model') != 'isothermal_adas' or report.get('kind') != 'stationary_scan_3d'
            or report.get('format_version') != 2 or report.get('stage') != 'refinement_complete'
            or report.get('parameters') != p or report.get('geometry') != signature['geometry']
            or report.get('model_options') != model_options
            or not isinstance(report.get('solutions'), list)):
        raise ValueError(f'Incomplete, old or mismatched case report: {path}')
    provenance = report['provenance']
    for key, expected in dict(code_sha256=signature['code_sha256'], runtime=signature['runtime'],
            source_config_sha256=signature['source_config_sha256'],
            data_sha256=_active_hashes(signature, density)).items():
        if provenance.get(key) != expected:
            raise ValueError(f'Case provenance mismatch ({key}): {path}')
    expected_grid = dict(T_eV=np.geomspace(options['temperature_min'], options['temperature_max'],
        options['temperature_count']).tolist(), j_target=(np.geomspace if options['j_grid'] == 'log'
        else np.linspace)(options['j_min'], options['j_max'], options['j_count']).tolist(),
        alpha=np.linspace(options['alpha_min'], options['alpha_max'], options['alpha_count']).tolist())
    if report['grid'] != expected_grid:
        raise ValueError('Inner grid differs from the batch settings.')
    for key in ('iterations', 'points', 'map_points', 'residual_tol', 'balance_tol',
                'dedup_tol', 'rtol', 'atol', 'max_step', 'max_starts', 'separation', 'j_grid'):
        if report['controls'][key] != options[key]:
            raise ValueError(f'Case control mismatch: {key}')
    shape = (options['alpha_count'], options['j_count'], options['temperature_count'])
    with np.load(path.parent/'residuals.npz', allow_pickle=False) as cube:
        if (cube['valid'].shape != shape or int(cube['valid'].sum()) != report['summary']['n_valid']
                or report['summary']['n_trials'] != int(np.prod(shape))):
            raise ValueError('Case map is missing or inconsistent.')
    identifiers = set()
    for root in report['solutions']:
        values = [root[key] for key in ('T_eV', 'theta', 'j_target', 'alpha', 'a_cm', 'mu_target', 'R_J', 'R_M', 'R_E')]
        if (not np.isfinite(values).all() or min(values[:3]) <= 0 or root['mu_target'] <= 0
                or not signature['geometry_g_min'] < root['alpha'] < 1
                or root.get('stationary_verified') is not True or root['solution_id'] in identifiers
                or not np.isclose(root['T_eV'], root['theta']*p['T_star_eV'], rtol=1e-10)
                or not np.isclose(root['a_cm'], root['alpha']*p['L_cm'], rtol=1e-10)):
            raise ValueError('Invalid/unverified root coordinates or normalization.')
        if not options['temperature_min']*(1-1e-10) <= root['T_eV'] <= options['temperature_max']*(1+1e-10):
            raise ValueError('Root temperature lies outside the refinement bounds.')
        identifiers.add(root['solution_id'])
        errors = [root['balances'][key] for key in ('particle_abs', 'momentum_abs', 'energy_abs')]
        if (not np.isfinite(errors).all() or min(errors) < 0 or max(errors) > options['balance_tol']
                or abs(root['R_J']) > options['residual_tol'] or abs(root['R_E']) > options['residual_tol']
                or abs(root['R_M']) > options['residual_tol']*(1+abs(root['mu_target']))):
            raise ValueError('Unverified boundary residual or integral balance.')
        budgets = root['balances']
        if not np.isfinite(list(budgets.values())).all():
            raise ValueError('Nonfinite integral budget.')
        if any(root['verification'][key] != options[key] for key in ('residual_tol', 'balance_tol')):
            raise ValueError('Root verification tolerances differ from the batch controls.')
        born, lost = (budgets[key] for key in ('particle_ionization_integral', 'particle_recombination_integral'))
        ion_energy = budgets['ionization_energy_loss_integral']
        mol_keys = ('particle_mar_integral', 'particle_mai_integral', 'mad_event_integral',
                    'mar_energy_loss_integral', 'mad_energy_loss_integral', 'mai_energy_loss_integral',
                    'molecular_momentum_loss_integral')
        if gas.n_d2_cm3 > 0:
            if any(key not in budgets or budgets[key] < 0 for key in mol_keys):
                raise ValueError('Missing or negative molecular budget.')
        elif any(budgets.get(key, 0.) != 0 for key in mol_keys):
            raise ValueError('Nonzero molecular budget without molecules.')
        net_source = born-lost+budgets.get('particle_mai_integral', 0.)-budgets.get('particle_mar_integral', 0.)
        if (not np.isfinite([born, lost, ion_energy]).all() or min(born, lost, ion_energy) < 0
                or abs(root['j_target']-1-net_source) > options['balance_tol']+abs(root['R_J'])
                or not np.isclose(budgets['particle_net_source_integral'], net_source, rtol=1e-10, atol=1e-15)
                or not np.isclose(ion_energy, IONIZATION_ENERGY_ERG/p['T_star_erg']*born, rtol=1e-10, atol=1e-15)
                or ((options['ionization_scale'] == 0 or gas.n_d_cm3 == 0) and born != 0)):
            raise ValueError('Inconsistent particle/ionization-energy counts.')
        if not np.isfinite(root['coefficients']['D']) or root['coefficients']['D'] < 0:
            raise ValueError('Invalid D coefficient.')
        ranges = root['coefficients']['local_ranges']
        expected_ranges = {'R_profile', 'Q_eff_profile', 'D_ion_profile', 'E_ion_profile', 'E_exc_profile'}
        if gas.n_d2_cm3 > 0:
            expected_ranges.update(MOLECULAR_PROFILE_FIELDS)
        if set(ranges) != expected_ranges:
            raise ValueError('Missing local coefficient ranges.')
        for key, limits in ranges.items():
            limits = np.asarray(limits, float)
            if (limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]
                    or (key != 'Q_eff_profile' and limits[0] < 0)):
                raise ValueError('Invalid local coefficient range.')
        profile_path = _inside(path.parent, root['profile_file'])
        with np.load(profile_path, allow_pickle=False) as profile:
            if any(not np.isfinite(profile[key]).all() for key in profile.files):
                raise ValueError('Nonfinite saved profile.')
            if (profile['z_cm'].ndim != 1 or len(profile['z_cm']) < 100
                    or np.any(np.diff(profile['z_cm']) <= 0) or np.any(profile['n_cm3'] <= 0)):
                raise ValueError('Invalid saved profile grid/density.')
            if gas.n_d2_cm3 > 0:
                for key in MOLECULAR_PROFILE_FIELDS:
                    field = key.removesuffix('_profile')
                    if (field not in profile or profile[field].shape != profile['z_cm'].shape
                            or np.any(profile[field] < 0)):
                        raise ValueError('Missing or invalid saved molecular coefficient profile.')
            for key, expected in ((key, root[key]) for key in ('T_eV', 'theta', 'alpha', 'j_target')):
                if not np.isclose(float(profile[key]), expected, rtol=1e-10, atol=0):
                    raise ValueError('Profile/root mismatch.')
            if (not np.isclose(profile['z_cm'][0], root['a_cm'], rtol=1e-10)
                    or not np.isclose(profile['z_cm'][-1], p['L_cm'], rtol=1e-10)
                    or not np.isclose(profile['u_over_cs'][-1], 1., rtol=1e-8)
                    or not np.isclose(profile['omega'][-1], lam*root['theta']*root['j_target'], rtol=1e-8)):
                raise ValueError('Profile endpoints do not match the wall conditions.')
    attempts = report['diagnostic']['attempts']
    if (len(attempts) != len(report['starts']) or report['diagnostic']['n_solutions'] != len(identifiers)
            or any((a['status'] == 'converged') != (a.get('converged_to') in identifiers) for a in attempts)):
        raise ValueError('Case refinement is incomplete or inconsistent.')
    return report


def _rows(case_id, relative, report):
    p, roots = report['parameters'], []
    common = dict(case_id=case_id, lambda_sh=p['lambda_sh'], n0_cm3=p['n0_cm3'],
                  **neutral_densities(p)._asdict())
    for root in report['solutions']:
        budgets = root['balances']
        row = dict(common, solution_id=case_id+'_'+root['solution_id'], local_solution_id=root['solution_id'],
            **{key: root[key] for key in ('T_eV', 'theta', 'a_cm', 'alpha', 'j_target', 'net_particle_change',
                'R_J', 'R_M', 'R_E', 'outside_scan_j_range', 'outside_scan_alpha_range')},
            **report['model_options'], D=root['coefficients']['D'],
            recombination_count_per_inlet=budgets['particle_recombination_integral'],
            ionization_count_per_inlet=budgets['particle_ionization_integral'], case_report=relative,
            profile_file=(Path(relative).parent/root['profile_file']).as_posix())
        for prefix in ('', 'cx_', 'excitation_', 'ionization_', 'effective_'):
            row[prefix+'energy_loss_fraction'] = budgets[prefix+'energy_loss_integral']
        for branch, count in (('mar', 'particle_mar_integral'), ('mad', 'mad_event_integral'),
                              ('mai', 'particle_mai_integral')):
            row[branch+'_count_per_inlet'] = budgets.get(count, 0.)
            row[branch+'_energy_loss_fraction'] = budgets.get(branch+'_energy_loss_integral', 0.)
        row.update({key+'_balance_abs': budgets[key+'_abs'] for key in ('particle', 'momentum', 'energy')})
        roots.append(row)
    summary = report['summary']
    counts = Counter(a['status'] for a in report['diagnostic']['attempts'])
    best = summary['nearest_sample_not_root']
    grid = dict(common, atomic_model=report['model_options']['atomic_model'],
        n_roots=len(roots), search_status='roots_found' if roots else 'no_root_found_in_finite_search',
        n_trials=summary['n_trials'], n_valid=summary['n_valid'], n_starts=len(report['starts']),
        n_converged_starts=counts['converged'], failed_start_count=len(report['starts'])-counts['converged'],
        best_sample_norm=None if best is None else best['residual_norm'],
        unresolved_numerical_count=summary['unresolved_numerical_count'],
        attempt_statuses=json.dumps(counts, sort_keys=True), case_report=relative)
    return roots, grid


def _save_tables(output, roots, grid):
    for name, records, fields in (('roots', roots, ROOT_FIELDS), ('grid', grid, GRID_FIELDS)):
        path = output/(name+'.csv')
        temporary = path.with_name(path.name+'.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';', extrasaction='raise')
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        temporary.replace(path)


def scan_lambda_neutral(lambdas, densities, config_path, output, *, geometry_path=None,
                        K=None, scan_options=None, resume=False, dry_run=False):
    """Return (roots,grid). Resume only completed pairs with identical provenance.

    Output must be new/empty, or an existing batch with resume=True. Unfinished
    attempts are preserved, then rerun in a new attempt directory. This is
    NOT continuation of physical branches or a proof that all roots exist.
    """
    lambdas = parameter_grid(lambdas, 'lambda_sh')
    densities = parameter_grid(densities, 'n0', allow_zero=True)
    options = _options(scan_options)
    args = single.build_parser().parse_args([])
    vars(args).update(options, config=Path(config_path).resolve(), geometry=geometry_path,
                      K=K, density=float(densities[-1]), lambda_sh=float(lambdas[0]))
    p, geometry, _, _, mode, provenance, T, js, alphas, _ = single._prepare(args)
    signature = dict(parameters=p, geometry=geometry.to_dict(), geometry_g_min=geometry.g_min,
        inlet_momentum=mode, lambdas=lambdas.tolist(), densities=densities.tolist(),
        options=options, **{key: provenance[key] for key in (
            'source_config_sha256', 'code_sha256', 'data_sha256', 'runtime')})
    signature_hash = hashlib.sha256(json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = Path(output).resolve()
    total = len(lambdas)*len(densities)
    print(f'Physical pairs: {total}; inner grid {len(T)} x {len(js)} x {len(alphas)}; '
          f'{total*len(T)*len(js)*len(alphas)} trial IVPs plus corrections.', flush=True)
    if dry_run:
        print('Dry run: no plasma calculations or output files.')
        return [], []
    manifest_path, checkpoint_path = output/'manifest.json', output/'checkpoint.json'
    if resume:
        manifest, checkpoint = _json(manifest_path), _json(checkpoint_path)
        if (manifest.get('model') != 'isothermal_adas' or manifest.get('kind') != 'lambda_neutral_scan'
                or manifest.get('format_version') != 1 or manifest.get('signature') != signature
                or checkpoint.get('signature_sha256') != signature_hash):
            raise ValueError('Resume mismatch: inputs, grids, code, ADAS data or runtime changed; use a new output.')
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Output is not empty. Use --resume for this batch or choose a new directory.')
        output.mkdir(parents=True, exist_ok=True)
        with (output/'source_config.json').open('xb') as stream:
            stream.write(args.config.read_bytes())
        single._save_json(output/'geometry.json', signature['geometry'])
        single._save_json(manifest_path, dict(format_version=1, model='isothermal_adas',
            kind='lambda_neutral_scan', signature=signature,
            warning='Finite independent searches, not an existence boundary or physical branch tracking.'))
        checkpoint = dict(format_version=1, signature_sha256=signature_hash, completed={})
        single._save_json(checkpoint_path, checkpoint)

    def check_environment():
        package = Path(__file__).parent
        if (code_hashes(options['atomic_model']) != signature['code_sha256']
                or {name: single._sha256(package/'processes_data'/name) for name in signature['data_sha256']}
                   != signature['data_sha256']
                or single._sha256(output/'source_config.json') != signature['source_config_sha256']
                or _json(output/'geometry.json') != signature['geometry']):
            raise ValueError('Code, ADAS data or frozen inputs changed during the batch; stopped.')

    pairs = {f'L{li:03d}_N{ni:03d}': (float(lam), float(density))
             for li, lam in enumerate(lambdas) for ni, density in enumerate(densities)}
    if set(checkpoint['completed'])-set(pairs):
        raise ValueError('Checkpoint contains unknown physical pairs.')
    roots, grid = [], []
    check_environment()
    for case_id, saved in checkpoint['completed'].items():
        path = _inside(output, saved['report'])
        if single._sha256(path) != saved['report_sha256']:
            raise ValueError('A completed case report was modified; cannot resume.')
        report = _read_case(path, *pairs[case_id], signature, options)
        for relative, checksum in saved['artifacts_sha256'].items():
            if single._sha256(_inside(path.parent, relative)) != checksum:
                raise ValueError('A completed case artifact was modified; cannot resume.')
        records, row = _rows(case_id, saved['report'], report)
        roots.extend(records)
        grid.append(row)
    _save_tables(output, roots, grid)  # Recover even if a previous CSV write was interrupted.
    status = dict(format_version=1, stage='running', n_completed=len(grid), n_total=total)
    single._save_json(output/'status.json', status)
    try:
        for case_id, (lam, density) in pairs.items():
            if case_id in checkpoint['completed']:
                continue
            check_environment()
            case = _inside(output, 'cases/'+case_id)
            case.mkdir(parents=True, exist_ok=True)
            index = 1
            while (case/f'attempt_{index:03d}').exists() or (case/f'attempt_{index:03d}.log').exists():
                index += 1
            attempt = case/f'attempt_{index:03d}'
            command = [sys.executable, '-B', '-u', '-m', 'isothermal_model_adas.scan_stationary',
                '--config', str(output/'source_config.json'), '--geometry', str(output/'geometry.json'),
                '--density', str(density), '--lambda-sh', str(lam), '--output', str(attempt),
                '--no-show', *_option_arguments(options)]
            status.update(active_case=case_id, lambda_sh=lam, n0_cm3=density)
            single._save_json(output/'status.json', status)
            print(f'Pair {len(grid)+1}/{total}: lambda={lam:g}, n0={density:.6e}', flush=True)
            code = _run_scan(command, Path(__file__).resolve().parents[1], case/f'attempt_{index:03d}.log')
            if code not in (0, 1):
                raise RuntimeError(f'{case_id}: child exit {code}; case not checkpointed. See its log.')
            path = attempt/'summary.json'
            report = _read_case(path, lam, density, signature, options)
            if code != (0 if report['solutions'] else 1):
                raise ValueError('Child exit code disagrees with its completed report.')
            relative = path.relative_to(output).as_posix()
            records, row = _rows(case_id, relative, report)
            artifacts = ['residuals.npz', *[root['profile_file'] for root in report['solutions']]]
            checkpoint['completed'][case_id] = dict(report=relative, report_sha256=single._sha256(path),
                artifacts_sha256={name: single._sha256(_inside(path.parent, name)) for name in artifacts})
            single._save_json(checkpoint_path, checkpoint)
            roots.extend(records)
            grid.append(row)
            _save_tables(output, roots, grid)
            status['n_completed'] = len(grid)
            single._save_json(output/'status.json', status)
        status.update(stage='complete', active_case=None)
        single._save_json(output/'status.json', status)
    except (Exception, KeyboardInterrupt) as error:
        status.update(stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=dict(type=type(error).__name__, message=str(error)))
        single._save_json(output/'status.json', status)
        raise
    return roots, grid
