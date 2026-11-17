"""Coefficient selection shared by the single, stationary and batch CLIs."""

from functools import lru_cache
import hashlib
from pathlib import Path

from .config import neutral_densities
from .cx_adas import ADASCX, make_D
from .exitation_adas import ADASExcitation, make_E_exc
from .ionization_adas import ADASIonization, make_ionization_coefficients
from .recombination_adas import ADASRecombination, make_adas_coefficients
from .sources import TemperatureOnly


ATOMIC_MODELS = ('hybrid', 'full-adas')
DEFAULT_ATOMIC_MODEL = 'hybrid'


def add_atomic_model_argument(parser):
    parser.add_argument('--atomic-model', choices=ATOMIC_MODELS,
        default=DEFAULT_ATOMIC_MODEL,
        help='hybrid: local ADAS ACD/PRB only; full-adas: also CCD, SCD and PLT.')


def build_atomic_coefficients(p, mode, rr_scale, ionization_scale, excitation_scale):
    """Return RR provider, other providers, active table objects and paths."""
    if mode not in ATOMIC_MODELS:
        raise ValueError(f'Unknown atomic model: {mode}')
    if neutral_densities(p).n_d2_cm3 > 0:
        raise ValueError('Molecular densities can be used to evaluate coefficients, but '
                         'MAR/MAD/MAI are not wired into transport yet. Set molecular_fraction=0 '
                         'for an atomic-only stationary solve.')
    rr = ADASRecombination() if rr_scale else None
    recombination = make_adas_coefficients(p, rr_scale, data=rr)
    tables = [rr] if rr is not None else []
    paths = [rr.acd_path, rr.prb_path] if rr is not None else []
    neutrals = p['n0_cm3'] > 0
    if mode == 'full-adas':
        cx = ADASCX() if neutrals else None
        ion = ADASIonization() if neutrals and ionization_scale else None
        exc = ADASExcitation() if neutrals and excitation_scale else None
        for table in (cx, ion, exc):
            if table is not None:
                tables.append(table)
                paths.append(table.path)
        providers = dict(D=make_D(p, data=cx),
            ionization=make_ionization_coefficients(p, ionization_scale, data=ion),
            excitation=make_E_exc(p, scale=excitation_scale, data=exc))
    elif not neutrals:
        providers = dict(D=0., ionization=(0., 0.), excitation=0.)
    else:
        from isothermal_cx_model import cx, excitation, ionization

        D = lru_cache(maxsize=512)(cx.make_D(p, mode='temperature'))
        ion = ionization.make_ionization_coefficients(p, ionization_scale)
        exc = excitation.make_E_exc(p, scale=excitation_scale)
        providers = dict(D=D, ionization=TemperatureOnly(ion) if callable(ion) else ion,
                         excitation=TemperatureOnly(exc) if callable(exc) else exc)
    return recombination, providers, tables, paths


def code_hashes(mode):
    """Include the reused legacy formulas in the hybrid resume fingerprint."""
    package = Path(__file__).resolve().parent
    paths = {path.name: path for path in sorted(package.glob('*.py'))}
    if mode == 'hybrid':
        for name in ('cx.py', 'ionization.py', 'excitation.py'):
            key = f'isothermal_cx_model/{name}'
            paths[key] = package.parent/key
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
