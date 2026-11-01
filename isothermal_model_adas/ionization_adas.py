"""Effective hydrogen ionization from SCD12_H; CGS, temperature in eV.

Coefficient layer: transport queries (theta,N) locally. H data approximate D.
SCD(T_e,n_e) is a collisional-radiative rate, not a ground-state-only cross
section. L_ion=I*SCD is the binding-energy cost for W excluding binding
energy; it is NOT full electron cooling. Excitation radiation needs PLT.
No plasma solve, data download or file loading takes place on import.
"""

from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .recombination_adas import (ADASDataError, ADASDomainError, EV_TO_ERG,
                                 IONIZATION_ENERGY_EV, IONIZATION_ENERGY_ERG)


SCD_PATH = Path(__file__).resolve().parent/'processes_data'/'scd12_h.dat'
_LOG_BOUND_TOL = 1e-8  # Printed endpoint rounding only; not extrapolation.


class ADASIonization:
    """One unresolved hydrogen SCD block, density-fast ADF11 layout.

    Log-bilinear interpolation in log10(T_e),log10(n_e),log10(SCD),
    consistent with our ACD/PRB implementation, not ADAS spline software.
    Both arguments are local plasma values; n_e is NOT neutral n0.
    Not a universal ADF11 reader. Caller must supply an SCD rate file.
    """
    def __init__(self, path=SCD_PATH):
        self.path = Path(path)
        lines = self.path.read_text(encoding='utf-8-sig').splitlines()
        try:
            charge, nd, nt, first, last = map(int, lines[0].split('/')[0].split())
            if (charge, first, last) != (1, 1, 1) or min(nd, nt) < 2 or 'HYDROGEN' not in lines[0].upper():
                raise ValueError('Expected one unresolved hydrogen ion stage.')
            blocks = [i for i, line in enumerate(lines) if 'IPRT=' in line.upper()]
            if len(blocks) != 1:
                raise ValueError('Expected exactly one coefficient block.')
            block = blocks[0]
            for name in ('IPRT', 'IGRD', 'Z1'):
                if int(re.search(r'\b'+name+r'\s*=\s*(\d+)', lines[block]).group(1)) != 1:
                    raise ValueError('Only IPRT=IGRD=Z1=1 is supported.')
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
            if 'UNITS OF CM**3 S-1' not in notice:
                raise ValueError('Expected SCD rate units cm^3/s.')
        except (ValueError, IndexError, AttributeError) as error:
            raise ADASDataError(f'{self.path}: {error}') from error
        for array in (log_T, log_n, log_values):
            array.setflags(write=False)
        self.log_T, self.log_n = log_T, log_n
        self.T_bounds_eV = tuple(10.**log_T[[0, -1]])
        self.n_bounds_cm3 = tuple(10.**log_n[[0, -1]])
        self._scd = RegularGridInterpolator((log_T, log_n), log_values,
                                             method='linear', bounds_error=True)

    def evaluate(self, T_eV, n_e_cm3):
        """Return SCD in cm^3/s; scalar or mutually broadcastable arrays."""
        T, n = np.broadcast_arrays(np.asarray(T_eV, float), np.asarray(n_e_cm3, float))
        if not np.isfinite(T).all() or not np.isfinite(n).all() or np.any(T <= 0) or np.any(n <= 0):
            raise ValueError('T_eV and n_e_cm3 must be finite and strictly positive.')
        coordinates = []
        for value, axis in ((T, self.log_T), (n, self.log_n)):
            logarithm = np.log10(value)
            if np.any(logarithm < axis[0]-_LOG_BOUND_TOL) or np.any(logarithm > axis[-1]+_LOG_BOUND_TOL):
                raise ADASDomainError(f'Outside SCD12_H: T_eV in {self.T_bounds_eV}, '
                                      f'n_e_cm3 in {self.n_bounds_cm3}; no extrapolation.')
            for endpoint in axis[[0, -1]]:
                logarithm = np.where(abs(logarithm-endpoint) <= _LOG_BOUND_TOL, endpoint, logarithm)
            coordinates.append(logarithm.ravel())
        points = np.column_stack(coordinates)
        result = (10.**self._scd(points)).reshape(T.shape)
        return float(result) if T.ndim == 0 else result


@lru_cache(maxsize=1)
def _default_data():
    return ADASIonization()


def k_ion(T_e_eV, n_e_cm3, *, data=None):
    """SCD(T_e,n_e) [cm^3/s]; both local temperature and density required."""
    return (data if data is not None else _default_data()).evaluate(T_e_eV, n_e_cm3)


def log_k_ion(T_e_eV, n_e_cm3, *, data=None):
    """Natural log of the numerical rate in cm^3/s, as in the old module.

    No extension below the SCD table, no artificial ionization cutoff.
    """
    value = np.log(k_ion(T_e_eV, n_e_cm3, data=data))
    return float(value) if np.ndim(value) == 0 else value


def L_ion(T_e_eV, n_e_cm3, *, data=None):
    """I*SCD [erg cm^3/s], binding-energy cost ONLY. No PLT or extra 3T/2."""
    return IONIZATION_ENERGY_ERG*k_ion(T_e_eV, n_e_cm3, data=data)


def ionization_coefficients(T_e_eV, n_e_cm3, *, data=None):
    """Return (SCD [cm^3/s], I*SCD [erg cm^3/s]) with one table query."""
    rate = k_ion(T_e_eV, n_e_cm3, data=data)
    return rate, IONIZATION_ENERGY_ERG*rate


def _positive_parameter(parameters, key):
    try:
        value = float(parameters[key])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Missing or invalid parameter: {key}') from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive.')
    return value


def make_ionization_coefficients(parameters, scale=0., n0_cm3=None, *, data=None):
    """Return (0,0) or coefficients(theta,N)->(D_ion,E_ion).

    D_ion = scale*L*n0/u_star * SCD(theta*T_star_eV,N*n_star).
    E_ion = (I/T_star_erg)*D_ion. Same scale in particles and energy.
    Source scales/n0 are copied; unlike the old function, the callable needs
    TWO arguments and must be evaluated locally, even for constant theta.
    Off scale=0 needs no inputs/table; n0=0 also needs no table/scales.
    """
    scale = float(scale)
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('Ionization scale must be finite and between 0 and 1.')
    if scale == 0:
        return (0., 0.)
    try:
        n0 = float(parameters['n0_cm3'] if n0_cm3 is None else n0_cm3)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('Missing or invalid neutral n0_cm3.') from error
    if not np.isfinite(n0) or n0 < 0:
        raise ValueError('Neutral n0_cm3 must be finite and nonnegative.')
    if n0 == 0:
        return (0., 0.)
    length, density, speed, energy, temperature = (_positive_parameter(parameters, key) for key in
        ('L_cm', 'n_star_cm3', 'U_star_cm_per_s', 'T_star_erg', 'T_star_eV'))
    if not np.isclose(energy, temperature*EV_TO_ERG, rtol=1e-10, atol=0):
        raise ValueError('T_star_erg and T_star_eV are inconsistent.')
    prefactor = scale*length*n0/speed
    epsilon = IONIZATION_ENERGY_ERG/energy
    if not np.isfinite([prefactor, epsilon]).all() or min(prefactor, epsilon) <= 0:
        raise ValueError('Ionization normalization is not representable.')
    tables = _default_data() if data is None else data

    def coefficients(theta, N):
        rate = k_ion(np.asarray(theta)*temperature, np.asarray(N)*density, data=tables)
        D_ion = prefactor*rate
        E_ion = epsilon*D_ion
        if not np.isfinite(D_ion).all() or not np.isfinite(E_ion).all():
            raise ValueError('Nonfinite dimensionless ionization coefficients.')
        return D_ion, E_ion

    return coefficients


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[.2, 1., 5., 20., 100.])
    parser.add_argument('--density', type=float, default=1e13, help='Local ELECTRON density [cm^-3], not n0.')
    parser.add_argument('--data', type=Path, default=SCD_PATH)
    args = parser.parse_args()
    try:
        rate, loss = ionization_coefficients(args.temperatures, args.density,
                                            data=ADASIonization(args.data))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Coefficients ONLY; no plasma solver. I*SCD excludes PLT radiation.')
    print('T_e [eV]       SCD [cm^3/s]       I*SCD [erg cm^3/s]')
    for T, k, L in zip(args.temperatures, rate, loss):
        print(f'{T:10.6g}  {k:18.9e}  {L:22.9e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
