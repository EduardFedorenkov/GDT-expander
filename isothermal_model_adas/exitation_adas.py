"""Excitation line cooling from PLT12_H; CGS, temperatures in eV.

Filename follows the requested exitation_adas spelling. Coefficient layer;
transport queries E_exc(theta,N) locally. PLT includes photon energy: convert
W cm^3 to erg cm^3/s, with no extra excitation/ionization energy multiplier.
Use instead of the old L_exc fit, not on top of it. I*SCD and PRB remain
separate. H data approximate D; escaping photons/local CR are assumptions.
No plasma solve, download or table loading happens on import.
"""

from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .recombination_adas import ADASDataError, ADASDomainError, EV_TO_ERG


PLT_PATH = Path(__file__).resolve().parent/'processes_data'/'plt12_h.dat'
_LOG_BOUND_TOL = 1e-8  # Printed endpoint rounding, NOT extrapolation.
WATT_TO_ERG_PER_S = 1e7


class ADASExcitation:
    """Read one unresolved hydrogen PLT12 block: IPRT=1, IGRD=0, Z1=1.

    Density-fast ADF11 layout. Log-bilinear interpolation, matching our
    ACD/SCD/PRB approach, not claiming to reproduce ADAS spline software.
    This is a PLT-specific reader, not a universal ADF11 reader.
    """
    def __init__(self, path=PLT_PATH):
        self.path = Path(path)
        lines = self.path.read_text(encoding='utf-8-sig').splitlines()
        try:
            charge, nd, nt, first, last = map(int, lines[0].split('/')[0].split())
            if (charge, first, last) != (1, 1, 1) or min(nd, nt) < 2 or 'HYDROGEN' not in lines[0].upper():
                raise ValueError('Expected one unresolved hydrogen ion stage.')
            blocks = [i for i, line in enumerate(lines) if 'IPRT=' in line.upper()]
            if len(blocks) != 1:
                raise ValueError('Expected exactly one PLT coefficient block.')
            block = blocks[0]
            tags = {name: int(re.search(r'\b'+name+r'\s*=\s*(\d+)', lines[block]).group(1))
                    for name in ('IPRT', 'IGRD', 'Z1')}
            if tags != {'IPRT': 1, 'IGRD': 0, 'Z1': 1}:
                raise ValueError('PLT12_H requires IPRT=1, IGRD=0, Z1=1.')
            grid = [float(v.replace('D', 'E')) for line in lines[2:block] for v in line.split()]
            if len(grid) != nd+nt:
                raise ValueError('Grid dimensions differ from the header.')
            values = []
            for line in lines[block+1:]:
                if line.lstrip().startswith(('C', 'c', '---')):
                    break
                values.extend(float(v.replace('D', 'E')) for v in line.split())
            if len(values) != nd*nt:
                raise ValueError('Coefficient count differs from the header.')
            log_n, log_T = np.array(grid[:nd]), np.array(grid[nd:])
            log_values = np.array(values).reshape(nt, nd)
            if (not np.isfinite(grid+values).all() or np.any(np.diff(log_n) <= 0)
                    or np.any(np.diff(log_T) <= 0)):
                raise ValueError('Need finite data and strictly increasing grids.')
            notice = ' '.join(' '.join(lines).upper().split())
            if 'RADIATED POWERS IN UNITS OF W CM**3' not in notice:
                raise ValueError('Expected PLT radiated-power units W cm^3.')
        except (ValueError, IndexError, AttributeError) as error:
            raise ADASDataError(f'{self.path}: {error}') from error
        for array in (log_T, log_n, log_values):
            array.setflags(write=False)
        self.log_T, self.log_n = log_T, log_n
        self.T_bounds_eV = tuple(10.**log_T[[0, -1]])
        self.n_bounds_cm3 = tuple(10.**log_n[[0, -1]])
        self._plt = RegularGridInterpolator((log_T, log_n), log_values,
                                             method='linear', bounds_error=True)

    def evaluate(self, T_eV, n_e_cm3):
        """Return Lambda_PLT=1e7*PLT [erg cm^3/s], scalar or broadcast arrays."""
        T, n = np.broadcast_arrays(np.asarray(T_eV, float), np.asarray(n_e_cm3, float))
        if not np.isfinite(T).all() or not np.isfinite(n).all() or np.any(T <= 0) or np.any(n <= 0):
            raise ValueError('T_eV and n_e_cm3 must be finite and strictly positive.')
        coordinates = []
        for value, axis in ((T, self.log_T), (n, self.log_n)):
            logarithm = np.log10(value)
            if np.any(logarithm < axis[0]-_LOG_BOUND_TOL) or np.any(logarithm > axis[-1]+_LOG_BOUND_TOL):
                raise ADASDomainError(f'Outside PLT12_H: T_eV in {self.T_bounds_eV}, '
                                      f'n_e_cm3 in {self.n_bounds_cm3}; no extrapolation.')
            for endpoint in axis[[0, -1]]:
                logarithm = np.where(abs(logarithm-endpoint) <= _LOG_BOUND_TOL, endpoint, logarithm)
            coordinates.append(logarithm.ravel())
        points = np.column_stack(coordinates)
        result = (WATT_TO_ERG_PER_S*10.**self._plt(points)).reshape(T.shape)
        return float(result) if T.ndim == 0 else result


@lru_cache(maxsize=1)
def _default_data():
    return ADASExcitation()


def L_exc(T_eV, n_e_cm3, *, data=None):
    """Excitation line-cooling coefficient [erg cm^3/s], not power density.

    P_exc=n_e*n0*L_exc(T_e,n_e). No extra photon energy, SCD cost or PRB.
    """
    return (data if data is not None else _default_data()).evaluate(T_eV, n_e_cm3)


def validate_excitation_scale(scale):
    try:
        scale = float(scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('excitation_scale must be a finite number in [0,1].') from error
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('excitation_scale must be finite and between 0 and 1.')
    return scale


def _positive_parameter(parameters, key):
    try:
        value = float(parameters[key])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Missing or invalid parameter: {key}') from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive.')
    return value


def make_E_exc(parameters, n0_cm3=None, scale=0., *, data=None):
    """Return 0 or local E(theta,N), N=n_e/n_star; same scale convention as old.

    E=scale*L*n0/(u_star*T_star_erg)*Lambda_PLT(theta*T_star_eV,N*n_star).
    Source scales and scalar n0 are snapshotted. Default off needs no source
    parameters/data; n0=0 also needs no table/scales. Callback has TWO args,
    unlike the old theta-only excitation fit; temperature may be constant,
    but density is local. Scalar/array broadcasting supported.
    """
    scale = validate_excitation_scale(scale)
    if scale == 0:
        return 0.
    try:
        n0 = float(parameters['n0_cm3'] if n0_cm3 is None else n0_cm3)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('Missing or invalid neutral n0_cm3.') from error
    if not np.isfinite(n0) or n0 < 0:
        raise ValueError('Neutral n0_cm3 must be finite and nonnegative.')
    if n0 == 0:
        return 0.
    length, density, speed, energy, temperature = (_positive_parameter(parameters, key) for key in
        ('L_cm', 'n_star_cm3', 'U_star_cm_per_s', 'T_star_erg', 'T_star_eV'))
    if not np.isclose(energy, temperature*EV_TO_ERG, rtol=1e-10, atol=0):
        raise ValueError('T_star_erg and T_star_eV are inconsistent.')
    prefactor = scale*length*n0/(speed*energy)
    if not np.isfinite(prefactor) or prefactor <= 0:
        raise ValueError('Excitation normalization is not representable.')
    tables = _default_data() if data is None else data

    def coefficient(theta, N):
        loss = L_exc(np.asarray(theta)*temperature, np.asarray(N)*density, data=tables)
        result = prefactor*loss
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite dimensionless excitation coefficient.')
        return result

    return coefficient


def E_exc(theta, N, parameters, n0_cm3=None, *, data=None):
    """Unit-scale E_exc(theta,N). Prefer make_E_exc for repeated RHS calls.

    This one-shot helper has explicit N as a new required argument.
    Returns scalar/array zeros without reading data for n0=0.
    """
    theta, N = np.broadcast_arrays(np.asarray(theta, float), np.asarray(N, float))
    if not np.isfinite(theta).all() or not np.isfinite(N).all() or np.any(theta < 0) or np.any(N < 0):
        raise ValueError('theta,N must be finite and nonnegative.')
    provider = make_E_exc(parameters, n0_cm3=n0_cm3, scale=1., data=data)
    if not callable(provider):
        return 0. if theta.ndim == 0 else np.zeros_like(theta)
    return provider(theta, N)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[.2, 1., 5., 20., 100.])
    parser.add_argument('--density', type=float, default=1e13, help='Local ELECTRON density [cm^-3], not neutral n0.')
    parser.add_argument('--data', type=Path, default=PLT_PATH)
    args = parser.parse_args()
    try:
        loss = L_exc(args.temperatures, args.density, data=ADASExcitation(args.data))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Coefficients ONLY; no plasma solver. Lambda_PLT=1e7*PLT; no I*SCD or PRB.')
    print('T_e [eV]      Lambda_PLT [erg cm^3/s]')
    for T, value in zip(args.temperatures, loss):
        print(f'{T:10.6g}  {value:24.9e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
