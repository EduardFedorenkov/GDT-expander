"""AMJUEL H.4 rates for the complete MAR/MAD/MAI branches via H2+.

Hydrogen fits used as a baseline for deuterium. Inputs are local electron
temperature (eV) and density (cm^-3); rates are in cm^3/s. No data are read
on import. These fits already include ion conversion and branching.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


AMJUEL_PATH = Path(__file__).resolve().parent / 'processes_data' / 'amjuel_molecular_h4.json'
_REACTIONS = (('MAR', '3.2.3r'), ('MAD', '3.2.3d'), ('MAI', '3.2.3i'))
_DENSITY_SCALE_CM3 = 1e8


class AMJUELDataError(ValueError):
    """Invalid AMJUEL coefficient file."""


class AMJUELDomainError(ValueError):
    """Inputs outside a documented fit range or rates not representable."""


class MolecularRates(NamedTuple):
    mar: float | np.ndarray
    mad: float | np.ndarray
    mai: float | np.ndarray


def _bounds(values):
    bounds = np.asarray(values, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] < bounds[1]):
        raise ValueError('Expected two finite, positive, increasing bounds.')
    return tuple(map(float, bounds))


class AMJUELMolecular:
    """Reader for the project's three AMJUEL H.4 matrices, not a general parser."""

    def __init__(self, path=AMJUEL_PATH):
        self.path = Path(path)
        try:
            doc = json.loads(self.path.read_text(encoding='utf-8-sig'))
            fit = doc['fit']
            if (doc['source']['database'] != 'AMJUEL' or doc['source']['section'] != 'H.4'
                    or fit['rate_unit'] != 'cm^3 s^-1'
                    or fit['coefficient_axes'] != ['temperature_log_power', 'density_log_power']
                    or fit['x'] != 'ln(T_e / 1 eV)' or fit['y'] != 'ln(n_e / 1e8 cm^-3)'):
                raise ValueError('Unexpected AMJUEL H.4 units, axes or log convention.')
            self.n_bounds_cm3 = _bounds(fit['density_range_cm3'])
            limits = fit['temperature_range_eV']
            self.T_bounds_eV = None if limits is None else _bounds(limits)
            matrices = []
            for name, reaction_id in _REACTIONS:
                record = doc['reactions'][name]
                if record['reaction'] != reaction_id:
                    raise ValueError(f'{name}: expected reaction {reaction_id}.')
                matrix = np.asarray(record['b'], dtype=float)
                if matrix.shape != (9, 9) or not np.isfinite(matrix).all():
                    raise ValueError(f'{name}: expected a finite 9 by 9 matrix.')
                matrix.setflags(write=False)
                matrices.append(matrix)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise AMJUELDataError(f'{self.path}: {error}') from error
        self._matrices = tuple(matrices)

    def evaluate(self, T_eV, n_e_cm3):
        """Return (K_MAR, K_MAD, K_MAI), scalar or broadcastable arrays."""
        T, n = np.broadcast_arrays(np.asarray(T_eV, float), np.asarray(n_e_cm3, float))
        if (not np.isfinite(T).all() or not np.isfinite(n).all()
                or np.any(T <= 0) or np.any(n <= 0)):
            raise ValueError('T_eV and n_e_cm3 must be finite and strictly positive.')
        for values, bounds, name in ((T, self.T_bounds_eV, 'T_eV'),
                                     (n, self.n_bounds_cm3, 'n_e_cm3')):
            if bounds is not None and np.any((values < bounds[0]) | (values > bounds[1])):
                raise AMJUELDomainError(f'Outside AMJUEL H.4: {name} in {bounds}.')
        # JSON rows are powers of log(T); columns are powers of log(n/1e8).
        x, y = np.log(T), np.log(n / _DENSITY_SCALE_CM3)
        rates = []
        for (name, _), matrix in zip(_REACTIONS, self._matrices):
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                rate = np.exp(np.polynomial.polynomial.polyval2d(x, y, matrix))
            if not np.isfinite(rate).all() or np.any(rate <= 0):
                raise AMJUELDomainError(f'{name}: rate is not finite and positive at this T_e and n_e.')
            rates.append(float(rate) if T.ndim == 0 else rate)
        return MolecularRates(*rates)


@lru_cache(maxsize=1)
def _default_data():
    return AMJUELMolecular()


def molecular_coefficients(T_eV, n_e_cm3, *, data=None):
    """Return full-branch rate coefficients (MAR, MAD, MAI) in cm^3/s."""
    return (data if data is not None else _default_data()).evaluate(T_eV, n_e_cm3)


def k_mar(T_eV, n_e_cm3, *, data=None):
    return molecular_coefficients(T_eV, n_e_cm3, data=data).mar


def k_mad(T_eV, n_e_cm3, *, data=None):
    return molecular_coefficients(T_eV, n_e_cm3, data=data).mad


def k_mai(T_eV, n_e_cm3, *, data=None):
    return molecular_coefficients(T_eV, n_e_cm3, data=data).mai


def _parameter(value, name, *, allow_zero=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number.') from error
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}.')
    return value


def make_molecular_coefficients(parameters, scale=0., n_d2_v0_cm3=None, *, data=None):
    """Return (0,0,0) or a callback (theta,N)->(D_MAR,D_MAD,D_MAI).

    For a constant ground-state molecular density, D_X = scale*L*n_D2(v=0)
    /u_star * K_X(theta*T_star_eV, N*n_star_cm3). N is the local plasma
    density normalized by n_star; it is not the molecular density.
    """
    scale = _parameter(scale, 'Molecular scale', allow_zero=True)
    if scale > 1:
        raise ValueError('Molecular scale must be between 0 and 1.')
    if scale == 0:
        return MolecularRates(0., 0., 0.)
    key = 'n_d2_v0_cm3'
    n_d2_v0_cm3 = _parameter(
        parameters.get(key) if n_d2_v0_cm3 is None else n_d2_v0_cm3, key, allow_zero=True)
    if n_d2_v0_cm3 == 0:
        return MolecularRates(0., 0., 0.)
    length, speed, temperature, density = (_parameter(parameters.get(key), key) for key in
        ('L_cm', 'U_star_cm_per_s', 'T_star_eV', 'n_star_cm3'))
    prefactor = _parameter(scale * length * n_d2_v0_cm3 / speed, 'Molecular normalization')
    tables = data if data is not None else _default_data()

    def coefficients(theta, N):
        rates = tables.evaluate(np.asarray(theta) * temperature, np.asarray(N) * density)
        values = MolecularRates(*(prefactor * rate for rate in rates))
        if any(not np.isfinite(value).all() for value in values):
            raise AMJUELDomainError('Nonfinite dimensionless molecular coefficients.')
        return values

    return coefficients


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1., 5., 20.])
    parser.add_argument('--density', type=float, default=1e13, help='Electron density [cm^-3].')
    parser.add_argument('--data', type=Path, default=AMJUEL_PATH)
    args = parser.parse_args()
    try:
        data = AMJUELMolecular(args.data)
        rates = data.evaluate(args.temperatures, args.density)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if data.T_bounds_eV is None:
        print('AMJUEL source does not specify explicit temperature bounds for these fits.')
    print(f'Hydrogen AMJUEL fits used as a D proxy; n_e = {args.density:.6g} cm^-3')
    print('T_e [eV]           K_MAR              K_MAD              K_MAI  [cm^3/s]')
    for temperature, mar, mad, mai in zip(args.temperatures, *rates):
        print(f'{temperature:10.6g} {mar:18.9e} {mad:18.9e} {mai:18.9e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
