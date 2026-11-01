"""CCD96_D charge-exchange rate and D_cx; CGS, temperature argument in eV.

Coefficient layer; transport receives make_D's scalar/D(theta). k_model(T)=CCD(T) is
an explicit approximation, not an assertion of identical cold-neutral
averaging to the old cx.py. No cross section or arbitrary-drift rate can
be recovered from this one-temperature table. For D+ + D the CX particle
source is zero; this is NOT an extra ACD recombination channel.
"""

from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from .recombination_adas import ADASDataError, ADASDomainError


CCD_PATH = Path(__file__).resolve().parent/'processes_data'/'ccd96_d.dat'
_LOG_BOUND_TOL = 1e-8  # Printed log-grid rounding, NOT physical extrapolation.


class ADASCX:
    """Strict reader of single-stage, density-independent CCD96_D data.

    Checks density independence rather than silently selecting a column.
    Interpolates log10(rate) linearly against log10(T); no extrapolation.
    Not a universal ADF11 reader. Coefficients have units cm^3/s.
    """
    def __init__(self, path=CCD_PATH):
        self.path = Path(path)
        lines = self.path.read_text(encoding='utf-8-sig').splitlines()
        try:
            charge, nd, nt, first, last = map(int, lines[0].split('/')[0].split())
            if (charge, first, last) != (1, 1, 1) or min(nd, nt) < 2 or 'DEUTERIUM' not in lines[0].upper():
                raise ValueError('Expected one deuterium ion stage and >=2 grid nodes.')
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
            table = np.array(values).reshape(nt, nd)
            if (not np.isfinite(grid+values).all() or np.any(np.diff(log_n) <= 0)
                    or np.any(np.diff(log_T) <= 0)):
                raise ValueError('Need finite data and ordered, distinct grid nodes.')
            if np.any(table != table[:, :1]):
                raise ValueError('Density-dependent CCD is not supported by k_cx(T).')
            notice = ' '.join(' '.join(lines).upper().split())
            if 'UNITS OF CM**3 S-1' not in notice:
                raise ValueError('Expected rate units cm^3/s.')
        except (ValueError, IndexError, AttributeError) as error:
            raise ADASDataError(f'{self.path}: {error}') from error
        self.log_T, self.log_n, self.log_k = log_T, log_n, table[:, 0].copy()
        for array in (self.log_T, self.log_n, self.log_k):
            array.setflags(write=False)
        self.T_bounds_eV = tuple(10.**self.log_T[[0, -1]])
        self.n_bounds_cm3 = tuple(10.**self.log_n[[0, -1]])

    def evaluate(self, T_eV):
        """Return scalar/array CCD(T), cm^3/s. No electron-density argument."""
        T = np.asarray(T_eV, dtype=float)
        if not np.isfinite(T).all() or np.any(T <= 0):
            raise ValueError('T_eV must be finite and strictly positive.')
        log_T = np.log10(T)
        if np.any(log_T < self.log_T[0]-_LOG_BOUND_TOL) or np.any(log_T > self.log_T[-1]+_LOG_BOUND_TOL):
            raise ADASDomainError(f'T_eV outside CCD96_D range {self.T_bounds_eV}; no extrapolation.')
        for endpoint in self.log_T[[0, -1]]:
            log_T = np.where(abs(log_T-endpoint) <= _LOG_BOUND_TOL, endpoint, log_T)
        value = 10.**np.interp(log_T, self.log_T, self.log_k)
        return float(value) if T.ndim == 0 else value


@lru_cache(maxsize=1)
def _default_data():
    return ADASCX()


def k_cx(T_i_eV, u_z_cms=0.0, *, data=None):
    """Model rate CCD(T_i_eV), cm^3/s; scalars/arrays supported.

    The old zero-drift calling convention is accepted for compatibility.
    A nonzero drift raises, never ignored or folded into an ad hoc T_eff.
    The table's neutral-temperature averaging still requires clarification.
    """
    u = np.asarray(u_z_cms, dtype=float)
    if u.ndim != 0 or not np.isfinite(u) or u != 0:
        raise ValueError('CCD96_D has no drift argument; require scalar u_z_cms=0.')
    return (data if data is not None else _default_data()).evaluate(T_i_eV)


def _parameter(parameters, key, *, nonnegative=False):
    try:
        value = float(parameters[key])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Missing or invalid parameter: {key}') from error
    if not np.isfinite(value) or value < 0 or (value == 0 and not nonnegative):
        raise ValueError(f'{key} must be finite and '+('nonnegative.' if nonnegative else 'positive.'))
    return value


def _neutral_density(parameters, override):
    source = parameters if override is None else {'n0_cm3': override}
    return _parameter(source, 'n0_cm3', nonnegative=True)


def D_cx(theta, parameters, n0_cm3=None, *, data=None):
    """D(theta) = L*n0/u_star * CCD(theta*T_star_eV).

    n0 is NEUTRAL density, not the electron-density grid from CCD. No table
    is needed if n0=0. theta may be an array; source scales and n0 scalar.
    """
    theta = np.asarray(theta, dtype=float)
    if not np.isfinite(theta).all() or np.any(theta < 0):
        raise ValueError('theta must be finite and nonnegative.')
    length = _parameter(parameters, 'L_cm')
    speed = _parameter(parameters, 'U_star_cm_per_s')
    n0 = _neutral_density(parameters, n0_cm3)
    if n0 == 0:
        return 0. if theta.ndim == 0 else np.zeros_like(theta)
    temperature = _parameter(parameters, 'T_star_eV')
    with np.errstate(over='raise', invalid='raise'):
        value = length*n0/speed*k_cx(theta*temperature, data=data)
    if not np.isfinite(value).all():
        raise ValueError('Unrepresentable D_cx normalization.')
    return float(value) if theta.ndim == 0 else value


def make_D(parameters, mode='temperature', n0_cm3=None, *, data=None):
    """Return number D or a callable D(theta), as in the old cx.py.

    temperature (DEFAULT): CCD(theta*T_star), snapshot source scales.
    constant: use explicit k_cx_cm3_per_s from parameters, NOT ADAS.
    No data loaded in constant/off modes. Arrays supported by D(theta).
    """
    if mode not in ('constant', 'temperature'):
        raise ValueError("mode must be 'constant' or 'temperature'.")
    n0 = _neutral_density(parameters, n0_cm3)
    length = _parameter(parameters, 'L_cm')
    speed = _parameter(parameters, 'U_star_cm_per_s')
    if n0 == 0:
        return 0.
    if mode == 'constant':
        value = length*n0/speed*_parameter(parameters, 'k_cx_cm3_per_s', nonnegative=True)
        if not np.isfinite(value):
            raise ValueError('Unrepresentable constant D normalization.')
        return float(value)
    snapshot = dict(L_cm=length, U_star_cm_per_s=speed,
                    T_star_eV=_parameter(parameters, 'T_star_eV'), n0_cm3=n0)
    tables = _default_data() if data is None else data

    def coefficient(theta):
        return D_cx(theta, snapshot, data=tables)

    return coefficient


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperatures', type=float, nargs='+', default=[.2, 1., 5., 20., 100.])
    parser.add_argument('--data', type=Path, default=CCD_PATH)
    args = parser.parse_args()
    try:
        values = k_cx(args.temperatures, data=ADASCX(args.data))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Coefficients only; no plasma solver. k_model(T)=CCD(T) approximation.')
    print('T [eV]       k_cx [cm^3/s]')
    for T, value in zip(args.temperatures, values):
        print(f'{T:10.6g}  {value:.9e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
