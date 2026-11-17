"""Coefficient selection shared by the single, stationary and batch CLIs."""

from functools import lru_cache
import hashlib
from pathlib import Path

from .config import neutral_densities
from .cx_adas import ADASCX, make_D
from .exitation_adas import ADASExcitation, make_E_exc
from .ionization_adas import ADASIonization, make_ionization_coefficients
from .molecular_amjuel import AMJUELMolecular, make_molecular_coefficients
from .recombination_adas import ADASRecombination, make_adas_coefficients
from .sources import TemperatureOnly


ATOMIC_MODELS = ('hybrid', 'full-adas')
DEFAULT_ATOMIC_MODEL = 'hybrid'


def add_atomic_model_argument(parser):
    parser.add_argument('--atomic-model', choices=ATOMIC_MODELS,
        default=DEFAULT_ATOMIC_MODEL,
        help='hybrid: local ADAS ACD/PRB only; full-adas: also CCD, SCD and PLT.')


def build_atomic_coefficients(p, mode, rr_scale, ionization_scale, excitation_scale):
    """Return RR, other local sources, active atomic/molecular tables and paths."""
    if mode not in ATOMIC_MODELS:
        raise ValueError(f'Unknown atomic model: {mode}')
    gas = neutral_densities(p)
    atomic_p = dict(p, n0_cm3=gas.n_d_cm3)
    rr = ADASRecombination() if rr_scale else None
    recombination = make_adas_coefficients(p, rr_scale, data=rr)
    tables = [rr] if rr is not None else []
    paths = [rr.acd_path, rr.prb_path] if rr is not None else []
    neutrals = gas.n_d_cm3 > 0
    if mode == 'full-adas':
        cx = ADASCX() if neutrals else None
        ion = ADASIonization() if neutrals and ionization_scale else None
        exc = ADASExcitation() if neutrals and excitation_scale else None
        for table in (cx, ion, exc):
            if table is not None:
                tables.append(table)
                paths.append(table.path)
        providers = dict(D=make_D(atomic_p, data=cx),
            ionization=make_ionization_coefficients(atomic_p, ionization_scale, data=ion),
            excitation=make_E_exc(atomic_p, scale=excitation_scale, data=exc))
    elif not neutrals:
        providers = dict(D=0., ionization=(0., 0.), excitation=0.)
    else:
        from isothermal_cx_model import cx, excitation, ionization

        D = lru_cache(maxsize=512)(cx.make_D(atomic_p, mode='temperature'))
        ion = ionization.make_ionization_coefficients(atomic_p, ionization_scale)
        exc = excitation.make_E_exc(atomic_p, scale=excitation_scale)
        providers = dict(D=D, ionization=TemperatureOnly(ion) if callable(ion) else ion,
                         excitation=TemperatureOnly(exc) if callable(exc) else exc)
    if gas.n_d2_cm3 > 0:
        table = AMJUELMolecular()
        # First approximation: ground-state molecular density = molecular density.
        providers.update(molecular=make_molecular_coefficients(
            p, scale=1., n_d2_v0_cm3=gas.n_d2_cm3, data=table), T_star_eV=p['T_star_eV'])
        tables.append(table)
        paths.append(table.path)
    return recombination, providers, tables, paths


def coefficient_temperature_bounds(tables, *, nominal_adas=False):
    """Intersection of documented temperature ranges, or None if none are known."""
    ranges = [(.2, 1e4) if nominal_adas and not isinstance(table, AMJUELMolecular)
              else table.T_bounds_eV for table in tables]
    ranges = [bounds for bounds in ranges if bounds is not None]
    if not ranges:
        return None
    lo, hi = max(bounds[0] for bounds in ranges), min(bounds[1] for bounds in ranges)
    if not lo < hi:
        raise ValueError('Active coefficient temperature ranges do not overlap.')
    return lo, hi


def code_hashes(mode):
    """Include the reused legacy formulas in the hybrid resume fingerprint."""
    package = Path(__file__).resolve().parent
    paths = {path.name: path for path in sorted(package.glob('*.py'))}
    if mode == 'hybrid':
        for name in ('cx.py', 'ionization.py', 'excitation.py'):
            key = f'isothermal_cx_model/{name}'
            paths[key] = package.parent/key
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
