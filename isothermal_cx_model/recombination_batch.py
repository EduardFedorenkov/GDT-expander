"""Batch orchestration of the checked 2D atomic scanner.

Each physical pair gets an independent full survey. No continuation-only
search, physical branch identification, or global completeness claim.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from simple_cx_model.config import load_cx_config
from .ionization import make_ionization_coefficients, IONIZATION_ENERGY_EV
from .scan_lambda import positive_grid, save_csv
from .excitation import validate_excitation_scale


ROOT_FIELDS = ('case_id', 'solution_id', 'local_solution_id', 'lambda_sh', 'n0_cm3',
    'format_version', 'cx_mode', 'rr_scale', 'excitation_scale', 'ionization_scale',
    'inlet_momentum', 'geometry', 'root_kind', 'T_eV', 'theta', 'a_cm', 'alpha',
    'j_target', 'net_particle_loss_fraction', 'recombination_count_per_inlet',
    'ionization_count_per_inlet', 'energy_ionization_fraction',
    'D', 'E_exc', 'R_rr', 'Q_rr', 'D_ion', 'E_ion', 'R_J', 'R_M', 'R_E',
    'particle_balance_abs', 'momentum_balance_abs', 'energy_balance_abs',
    'outside_scan_j_range', 'case_report')
GRID_FIELDS = ('case_id', 'lambda_sh', 'n0_cm3', 'cx_mode', 'rr_scale',
    'format_version', 'excitation_scale', 'ionization_scale', 'n_roots', 'search_status', 'n_trials', 'n_valid',
    'candidate_cell_count', 'n_starts', 'n_converged_starts', 'best_sample_norm',
    'unresolved_numerical_count', 'failed_start_count', 'scan_statuses',
    'attempt_statuses', 'case_report')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def code_fingerprint(project):
    """Conservative source fingerprint; excludes pycache, data and tests."""
    digest = hashlib.sha256()
    for package in ('isothermal_cx_model', 'simple_cx_model'):
        for path in sorted((Path(project)/package).rglob('*.py')):
            digest.update(path.relative_to(project).as_posix().encode('utf-8')+b'\0')
            digest.update(path.read_bytes()+b'\0')
    return digest.hexdigest()


def _run_scan(command, cwd, log_path):
    """No shell. Relay progress and save stdout/stderr; do not orphan a child."""
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    with Path(log_path).open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
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


def _read_case(path, lam, density, ionization_scale=0.):
    """A child exit code alone never becomes a root/no-root result."""
    report = _json(path)
    if report.get('ionization_scale') != float(ionization_scale):
        raise ValueError(f'Mismatched ionization scale in {path}')
    if (report.get('format_version') != 2 or report.get('stage') != 'refinement_complete'
            or report['parameters']['lambda_sh'] != float(lam)
            or report['parameters']['n0_cm3'] != float(density)
            or not isinstance(report.get('solutions'), list)
            or not isinstance(report.get('diagnostic'), dict)):
        raise ValueError(f'Incomplete or mismatched case report: {path}')
    identifiers = set()
    ionization = make_ionization_coefficients(report['parameters'], scale=ionization_scale)
    for root in report['solutions']:
        fields = ('theta', 'T_eV', 'alpha', 'a_cm', 'j_target', 'R_J', 'R_M', 'R_E')
        if (not np.isfinite([root[key] for key in fields]).all()
                or min(root['theta'], root['T_eV'], root['a_cm']) <= 0
                or not 0 < root['alpha'] < 1 or root['j_target'] <= 0
                or root['solution_id'] in identifiers):
            raise ValueError(f'Invalid root summary: {path}')
        identifiers.add(root['solution_id'])
        errors = [root['balances'][key] for key in ('particle_abs', 'momentum_abs', 'energy_abs')]
        balance_tol = root['verification']['balance_tol']
        residual_tol = root['verification']['residual_tol']
        if (not np.isfinite(errors+[balance_tol, residual_tol, root['mu_target']]).all()
                or min(balance_tol, residual_tol, root['mu_target']) <= 0 or min(errors) < 0
                or max(errors) > balance_tol):
            raise ValueError(f'Unverified integral balance in {path}')
        if (abs(root['R_J']) > residual_tol or abs(root['R_E']) > residual_tol
                or abs(root['R_M']) > residual_tol*(1+abs(root['mu_target']))):
            raise ValueError(f'Unverified inlet residual in {path}')
        coefficients = root['coefficients']
        values = [coefficients[key] for key in ('D', 'E_exc', 'R_rr', 'Q_rr', 'D_ion', 'E_ion')]
        expected = ionization(root['theta']) if callable(ionization) else ionization
        if (not np.isfinite(values).all() or min(values) < 0
                or not np.allclose(values[-2:], expected, rtol=1e-10, atol=0)):
            raise ValueError(f'Inconsistent ionization coefficients in {path}')
        if (not np.isclose(root['T_eV'], root['theta']*report['parameters']['T_star_eV'], rtol=1e-10, atol=0)
                or not np.isclose(root['a_cm'], root['alpha']*report['parameters']['L_cm'], rtol=1e-10, atol=0)):
            raise ValueError(f'Inconsistent source normalization in {path}')
        born = root['balances']['particle_ionization_integral']
        lost = root['balances']['particle_recombination_integral']
        ion_energy = root['balances']['energy_ionization_integral']
        if (not np.isfinite([born, lost, ion_energy]).all() or min(born, lost, ion_energy) < 0
                or abs(root['j_target']-1-(born-lost)) > balance_tol+abs(root['R_J'])
                or not np.isclose(ion_energy, IONIZATION_ENERGY_EV/report['parameters']['T_star_eV']*born,
                                  rtol=1e-10, atol=0)
                or not np.allclose([root['ionization_count_per_inlet'], root['recombination_count_per_inlet'],
                                    root['net_particle_loss_fraction']],
                                   [born, lost, 1-root['j_target']], rtol=1e-12, atol=1e-12)
                or (ionization_scale == 0 and born != 0)):
            raise ValueError(f'Inconsistent particle/ionization energy counts in {path}')
    return report


def _rows(case_id, report_path, report):
    common = dict(case_id=case_id, lambda_sh=report['parameters']['lambda_sh'],
        format_version=2, ionization_scale=report['ionization_scale'],
        n0_cm3=report['parameters']['n0_cm3'], cx_mode=report['cx_mode'],
        rr_scale=report['rr_scale'], excitation_scale=report['excitation_scale'],
        case_report=report_path)
    roots = []
    for root in report['solutions']:
        roots.append(dict(common, solution_id=case_id+'_'+root['solution_id'],
            local_solution_id=root['solution_id'], root_kind='verified_stationary_atomic',
            inlet_momentum=report['inlet_momentum'], geometry=report['geometry']['type'],
            **{key: root[key] for key in ('T_eV', 'theta', 'a_cm', 'alpha', 'j_target',
                'R_J', 'R_M', 'R_E', 'outside_scan_j_range')},
            **{key: root[key] for key in ('net_particle_loss_fraction',
                'recombination_count_per_inlet', 'ionization_count_per_inlet')},
            energy_ionization_fraction=root['balances']['energy_ionization_integral'],
            **root['coefficients'],
            particle_balance_abs=root['balances']['particle_abs'],
            momentum_balance_abs=root['balances']['momentum_abs'],
            energy_balance_abs=root['balances']['energy_abs']))
    summary = report['summary']
    attempts = report['diagnostic']['attempts']
    counts = Counter(item['status'] for item in attempts)
    nearest = summary['nearest_sample_not_root']
    grid = dict(common, n_roots=len(roots),
        search_status='roots_found' if roots else 'no_root_found_from_this_search',
        n_trials=summary['n_trials'], n_valid=summary['n_valid'],
        candidate_cell_count=summary['candidate_cell_count'], n_starts=len(report['starts']),
        n_converged_starts=counts['converged'],
        failed_start_count=len(attempts)-counts['converged'],
        best_sample_norm=None if nearest is None else nearest['log_residual_norm'],
        unresolved_numerical_count=summary['unresolved_numerical_count'],
        scan_statuses=json.dumps(summary['statuses'], sort_keys=True),
        attempt_statuses=json.dumps(counts, sort_keys=True))
    return roots, grid


def _save_tables(output, roots, grid, mode):
    for name, records, fields in [('roots', roots, ROOT_FIELDS), ('grid', grid, GRID_FIELDS)]:
        path = output/f'{name}_{mode}_recombination.csv'
        temporary = path.with_name(path.name+'.tmp')
        save_csv(temporary, records, fields)
        temporary.replace(path)


def _check_frozen(output, parameters, geometry_spec):
    if (load_cx_config(output/'source_config.json') != parameters
            or _json(output/'geometry.json') != geometry_spec):
        raise ValueError('Frozen source/geometry differs from the manifest; calculations stopped.')


def scan_lambda_neutral(lambdas, densities, config_path, output, *, geometry,
                       scan_arguments=(), ionization_scale=0., resume=False):
    """Return (roots, grid); save a completed-pair checkpoint and CSV after each.

    Runs the existing scan_recombination CLI in an isolated Python process.
    Its IVP, start selection, root verification and plots are not reimplemented.
    Interrupted/failed attempts are preserved; resume uses a NEW attempt dir.
    ionization_scale is batch-controlled, explicit in children and manifest.
    """
    lambdas, densities = positive_grid(lambdas, 'lambda'), positive_grid(densities, 'n0')
    ionization_scale = validate_excitation_scale(ionization_scale)
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    project = Path(__file__).resolve().parents[1]
    parameters = load_cx_config(config_path)
    source_text = config_path.read_text(encoding='utf-8-sig')
    arguments = list(map(str, scan_arguments))
    reserved = {'--config', '--output', '--density', '--lambda-sh', '--geometry',
                '--K', '--map-only', '--no-show', '--ionization-scale'}
    if any(item.startswith('--') and any(flag.startswith(item.split('=')[0]) for flag in reserved)
           for item in arguments):
        raise ValueError('scan_arguments overrides a batch-controlled argument.')
    signature = dict(format_version=2, ionization_scale=ionization_scale,
        lambdas=lambdas.tolist(), densities=densities.tolist(),
        parameters=parameters, geometry=geometry.to_dict(), scan_arguments=arguments,
        code_sha256=code_fingerprint(project))
    if resume:
        if not (output/'manifest.json').is_file() or _json(output/'manifest.json') != signature:
            raise ValueError('Resume requires the same source parameters, code, geometry and grids.')
        progress = _json(output/'progress.json') if (output/'progress.json').exists() else dict(completed={})
        if not isinstance(progress, dict) or not isinstance(progress.get('completed'), dict):
            raise ValueError('Invalid progress.json; no completed-pair information will be guessed.')
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Output must be absent/empty; use --resume only for a matching scan.')
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output/'manifest.json', signature)
        (output/'source_config.json').write_text(source_text, encoding='utf-8')
        _write_json(output/'geometry.json', geometry.to_dict())
        progress = dict(completed={})
    # Children read frozen inputs, not a user-edited JSON half-way through a scan.
    _check_frozen(output, parameters, signature['geometry'])
    roots, grid = [], []
    total = len(lambdas)*len(densities)
    case_index = 0
    for lam in lambdas:
        for density in densities:
            case_index += 1
            case_id = f'pair_{case_index:05d}'
            print(f'\n[{case_index}/{total}] lambda={lam:g}, n0={density:.6e}', flush=True)
            if code_fingerprint(project) != signature['code_sha256']:
                raise RuntimeError('Project code changed during this scan; stopped to avoid mixed models.')
            _check_frozen(output, parameters, signature['geometry'])
            if case_id in progress['completed']:
                relative = progress['completed'][case_id]['report']
                path = (output/relative).resolve()
                if not path.is_relative_to(output):
                    raise ValueError('Case report path escapes the output directory.')
                report = _read_case(path, lam, density, ionization_scale)
                print('Reusing completed pair.', flush=True)
            else:
                case_dir = output/'cases'/case_id
                attempt_index = 1
                while (case_dir/f'attempt_{attempt_index:03d}').exists():
                    attempt_index += 1
                attempt_dir = case_dir/f'attempt_{attempt_index:03d}'
                attempt_dir.mkdir(parents=True)
                result_dir = attempt_dir/'result'
                progress.update(stage='running', active_case=case_id)
                _write_json(output/'progress.json', progress)
                command = [sys.executable, '-B', '-m', 'isothermal_cx_model.scan_recombination',
                    '--config', str(output/'source_config.json'), '--geometry', str(output/'geometry.json'),
                    '--density', str(float(density)), '--lambda-sh', str(float(lam)),
                    '--output', str(result_dir), '--no-show',
                    '--ionization-scale', str(ionization_scale), *arguments]
                _write_json(attempt_dir/'command.json', command)
                try:
                    exit_code = _run_scan(command, project, attempt_dir/'scan.log')
                    if exit_code not in (0, 1):
                        raise RuntimeError(f'Child exit code {exit_code}. See {attempt_dir / "scan.log"}')
                    report = _read_case(result_dir/'summary.json', lam, density, ionization_scale)
                    if (exit_code == 0) != bool(report['solutions']):
                        raise RuntimeError('Child exit code conflicts with its solution report.')
                    if code_fingerprint(project) != signature['code_sha256']:
                        raise RuntimeError('Project code changed during the child calculation.')
                    _check_frozen(output, parameters, signature['geometry'])
                except (ValueError, OSError, KeyError, RuntimeError) as error:
                    progress.update(stage='failed', failure=str(error), active_case=case_id)
                    _write_json(output/'progress.json', progress)
                    raise RuntimeError(f'{case_id} failed; prior pairs saved. See {attempt_dir}') from error
                relative = (result_dir/'summary.json').relative_to(output).as_posix()
                progress['completed'][case_id] = dict(report=relative, exit_code=exit_code)
                progress.pop('failure', None)
                _write_json(output/'progress.json', progress)
            found, diagnostic = _rows(case_id, relative, report)
            roots.extend(found)
            grid.append(diagnostic)
            _save_tables(output, roots, grid, report['cx_mode'])
            print(f'Pair complete: {len(found)} verified roots; total retained: {len(roots)}.', flush=True)
    progress.update(stage='complete', active_case=None)
    _write_json(output/'progress.json', progress)
    return roots, grid
