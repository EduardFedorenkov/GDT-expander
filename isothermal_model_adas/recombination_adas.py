"""Hydrogen ADF11 ACD12/PRB12 coefficients; CGS, temperature in eV.

Standalone coefficient module, NOT connected to the transport solver.
ACD is effective recombination; PRB includes recombination AND bremsstrahlung.
For W excluding atomic binding energy: C_eff = 1e7*PRB - I*ACD.
Do not clip negative C_eff or log-interpolate this signed difference.
Hydrogen data are used as an approximation for D with I=13.6 eV.
"""

from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator


EV_TO_ERG = 1.602176634e-12
IONIZATION_ENERGY_EV = 13.6
IONIZATION_ENERGY_ERG = IONIZATION_ENERGY_EV * EV_TO_ERG
DATA_DIR = Path(__file__).resolve().parent/'processes_data'
ACD_PATH = DATA_DIR/'acd12_h.dat'
PRB_PATH = DATA_DIR/'prb12_h.dat'
# Allows nominal endpoints .2 eV, 5e7 and 2e15 cm^-3 despite printed log rounding.
# This is NOT permission to extrapolate to physically out-of-table states.
_LOG_BOUND_TOL = 1e-8


class ADASDataError(ValueError):
    """Malformed, incompatible, or unsupported ADF11 input data."""


class ADASDomainError(ValueError):
    """A positive (T_e,n_e) query lies outside the tabulated domain."""


def _read_hydrogen_table(path, kind):
    """Read the single, unresolved hydrogen block used by ACD12/PRB12.

    Not a general ADF11 parser: other elements, multiple stages/metastables,
    extra blocks and unrecognized units are rejected, not silently reused.
    Numeric layout: density grid, temperature grid, then density-fast rows.
    """
    path = Path(path)
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    try:
        header = tuple(map(int, lines[0].split('/')[0].split()))
        charge, nd, nt, first, last = header
        if (charge, first, last) != (1, 1, 1) or min(nd, nt) < 2 or 'HYDROGEN' not in lines[0].upper():
            raise ValueError('Expected one unresolved hydrogen ion stage.')
        blocks = [i for i, line in enumerate(lines) if 'IPRT=' in line.upper()]
        if len(blocks) != 1:
            raise ValueError('Expected exactly one coefficient block.')
        block = blocks[0]
        tags = {name: int(re.search(r'\b'+name+r'\s*=\s*(\d+)', lines[block]).group(1))
                for name in ('IPRT', 'IGRD', 'Z1')}
        if any(value != 1 for value in tags.values()):
            raise ValueError('Only IPRT=IGRD=Z1=1 is supported.')
        # All grids/coefficients in these files are base-10 logarithms.
        grid = [float(v.replace('D', 'E')) for line in lines[2:block] for v in line.split()]
        if len(grid) != nd+nt:
            raise ValueError('Grid count differs from the header.')
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
            raise ValueError('Need finite values and strictly increasing grids.')
        text = ' '.join(' '.join(lines).upper().split())
        unit = 'CM**3 S-1' if kind == 'ACD' else 'W CM**3'
        if 'UNITS OF '+unit not in text:
            raise ValueError(f'Unrecognized {kind} units; expected {unit}.')
    except (ValueError, IndexError, AttributeError) as error:
        raise ADASDataError(f'{path}: {error}') from error
    return log_T, log_n, log_values


class ADASRecombination:
    """Load a paired dataset once; log-bilinear interpolation, no extrapolation.

    Queries accept scalars or mutually broadcastable arrays. The interpolation
    is our explicit first approximation, not a claim to reproduce ADAS splines.
    Grid arrays are read-only; restart/reload after replacing the data files.
    """
    def __init__(self, acd_path=ACD_PATH, prb_path=PRB_PATH):
        self.acd_path, self.prb_path = Path(acd_path), Path(prb_path)
        T, n, acd = _read_hydrogen_table(self.acd_path, 'ACD')
        pT, pn, prb = _read_hydrogen_table(self.prb_path, 'PRB')
        if not np.array_equal(T, pT) or not np.array_equal(n, pn):
            raise ADASDataError('ACD and PRB grids must coincide for this paired dataset.')
        for array in (T, n, acd, prb):
            array.setflags(write=False)
        self.log_T, self.log_n = T, n
        self.T_bounds_eV = tuple(10.**T[[0, -1]])
        self.n_bounds_cm3 = tuple(10.**n[[0, -1]])
        self._acd = RegularGridInterpolator((T, n), acd, method='linear', bounds_error=True)
        self._prb = RegularGridInterpolator((T, n), prb, method='linear', bounds_error=True)

    def evaluate(self, T_eV, n_e_cm3):
        """Return (alpha [cm^3/s], L_gamma [erg cm^3/s], C_eff [erg cm^3/s])."""
        T, n = np.broadcast_arrays(np.asarray(T_eV, dtype=float), np.asarray(n_e_cm3, dtype=float))
        if not np.isfinite(T).all() or not np.isfinite(n).all() or np.any(T <= 0) or np.any(n <= 0):
            raise ValueError('T_eV and n_e_cm3 must be finite and strictly positive.')
        coordinates = []
        for value, axis in ((T, self.log_T), (n, self.log_n)):
            logarithm = np.log10(value)
            if np.any(logarithm < axis[0]-_LOG_BOUND_TOL) or np.any(logarithm > axis[-1]+_LOG_BOUND_TOL):
                raise ADASDomainError(f'Query outside ACD12/PRB12: T_eV in {self.T_bounds_eV}, '
                                      f'n_e_cm3 in {self.n_bounds_cm3}. No extrapolation.')
            for endpoint in (axis[0], axis[-1]):
                logarithm = np.where(abs(logarithm-endpoint) <= _LOG_BOUND_TOL, endpoint, logarithm)
            coordinates.append(logarithm.ravel())
        points = np.column_stack(coordinates)
        alpha = (10.**self._acd(points)).reshape(T.shape)
        photons = (1e7*10.**self._prb(points)).reshape(T.shape)
        effective = photons-IONIZATION_ENERGY_ERG*alpha
        if T.ndim == 0:
            return float(alpha), float(photons), float(effective)
        return alpha, photons, effective


@lru_cache(maxsize=1)
def _default_data():
    return ADASRecombination()


def adas_coefficients(T_eV, n_e_cm3, *, data=None):
    """Return (alpha_A, C_eff), like rr_coefficients but WITH local density.

    C_eff includes PRB bremsstrahlung and the -I*ACD correction. The unchanged
    ion kinetic/internal-energy removal alpha_A*(m*u^2/2+3*T_i/2) is separate.
    """
    alpha, _, effective = (data if data is not None else _default_data()).evaluate(T_eV, n_e_cm3)
    return alpha, effective


def alpha_adas(T_eV, n_e_cm3, *, data=None):
    return adas_coefficients(T_eV, n_e_cm3, data=data)[0]


def C_eff(T_eV, n_e_cm3, *, data=None):
    return adas_coefficients(T_eV, n_e_cm3, data=data)[1]


def L_gamma_adas(T_eV, n_e_cm3, *, data=None):
    """PRB photon coefficient in CGS, NOT the W-energy source by itself."""
    return (data if data is not None else _default_data()).evaluate(T_eV, n_e_cm3)[1]


def make_adas_coefficients(parameters, scale=0., *, data=None):
    """Return (0,0), or coefficients(theta,N) -> (R_A,Q_eff), N=n_e/n_star.

    Unlike the old factory, the callable has TWO arguments; it is NOT a drop-in
    replacement for theta -> (R,Q). R_A>=0 but Q_eff may be negative. No n0.
    scale multiplies the ENTIRE ACD/PRB channel, including bremsstrahlung.
    Off needs no parameters or files. Active factory snapshots source scales.
    """
    scale = float(scale)
    if not np.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('ADAS scale must be finite and between 0 and 1.')
    if scale == 0:
        return (0., 0.)
    length, density, speed, energy, temperature = (float(parameters[key]) for key in
        ('L_cm', 'n_star_cm3', 'U_star_cm_per_s', 'T_star_erg', 'T_star_eV'))
    if (not np.isfinite([length, density, speed, energy, temperature]).all()
            or min(length, density, speed, energy, temperature) <= 0
            or not np.isclose(energy, temperature*EV_TO_ERG, rtol=1e-10, atol=0)):
        raise ValueError('Need positive consistent CGS/eV source scales.')
    prefactor = scale*length*density/speed
    if not np.isfinite(prefactor) or prefactor <= 0:
        raise ValueError('ADAS normalization is not representable.')
    tables = _default_data() if data is None else data

    def coefficients(theta, N):
        alpha, effective = adas_coefficients(np.asarray(theta)*temperature,
                                             np.asarray(N)*density, data=tables)
        return prefactor*alpha, prefactor*effective/energy

    return coefficients


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[.2, 1., 5., 20., 100.])
    parser.add_argument('--density', type=float, default=1e13, help='Electron density [cm^-3], NOT n0.')
    args = parser.parse_args()
    try:
        tables = _default_data()
        alpha, photons, effective = tables.evaluate(args.temperatures, args.density)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Coefficients ONLY; no transport solver changed. PRB includes bremsstrahlung.')
    print('T_e [eV]    alpha_A [cm^3/s]    PRB [erg cm^3/s]    C_eff [erg cm^3/s]')
    for T, a, b, c in zip(args.temperatures, alpha, photons, effective):
        print(f'{T:10.6g}  {a:18.9e}  {b:18.9e}  {c:18.9e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
