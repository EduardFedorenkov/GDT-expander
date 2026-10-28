"""Direct electron-impact ionization of ground-state atomic D (H approximation).

Janev, Reiter, Samm (2003), Juel-4105, section 2.1.2, Eq. (14), Table 3:
https://www.eirene.de/Documentation/report_4105.pdf

Cross sections are for H(1s), used approximately for D(1s), with I=13.6 eV.
Stationary cold targets, zero-drift Maxwellian electrons, nonrelativistic
kinematics. No excited-state/stepwise ionization, molecules or hot tail.
This module supplies coefficients ONLY; no transport equations are changed.
"""

from functools import lru_cache
import math
import numpy as np
from scipy.integrate import quad


EV_TO_ERG = 1.602176634e-12
M_E_G = 9.1093837139e-28
IONIZATION_ENERGY_EV = 13.6
IONIZATION_ENERGY_ERG = IONIZATION_ENERGY_EV * EV_TO_ERG
_A = (0.18450, -0.032226, -0.034539, 1.4003, -2.8115, 2.2986)
_MAXWELL_FACTOR = math.sqrt(8 * EV_TO_ERG / (math.pi * M_E_G))
_LOG_PREFACTOR = math.log(_MAXWELL_FACTOR * 1e-13 / IONIZATION_ENERGY_EV**2)
_QUAD_RTOL = 1e-9


def _nonnegative_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f'{name} must be finite and nonnegative.')
    return array


def _polynomial(y):
    return _A[1] + y * (_A[2] + y * (_A[3] + y * (_A[4] + y * _A[5])))


def sigma_ion_D(E_eV):
    """Ionization cross section [cm^2]; electron collision energy in eV.

    H(1s) fit from Eq. (14)/Table 3, approximated for D. Exactly zero at and
    below the threshold, NOT below a threshold in electron temperature.
    Scalar or array; no deuteron-mass scaling and no CX center-of-mass factor.
    """
    energy = _nonnegative_array(E_eV, 'E_eV')
    result = np.zeros_like(energy)
    above = energy > IONIZATION_ENERGY_EV
    excess = energy[above] - IONIZATION_ENERGY_EV
    y = excess / energy[above]
    bracket = (_A[0] * np.log1p(excess / IONIZATION_ENERGY_EV)
               + y * _polynomial(y))
    result[above] = (1e-13 / IONIZATION_ENERGY_EV) * bracket / energy[above]
    return float(result) if result.ndim == 0 else result


def _scaled_bracket(x, tau):
    """Published bracket / tau, tau=T/I. Finite nonzero limit as T->0.

    For E=I+T*x, y=tau*x/(1+tau*x). Using the bracket divided by tau
    avoids a tiny quadrature amplitude at low T. log1p preserves threshold
    accuracy; the overflow branch only protects extreme numerical inputs.
    """
    delta = tau * x
    if math.isinf(delta):
        inverse = (1.0 / tau) / x
        return x * inverse * (_A[0] * (math.log(tau) + math.log(x)) + _polynomial(1.0))
    log_ratio = 1.0 if delta == 0 else math.log1p(delta) / delta
    y = delta / (1.0 + delta)
    return x * (_A[0] * log_ratio + _polynomial(y) / (1.0 + delta))


@lru_cache(maxsize=512)
def _log_k_single(temperature):
    if temperature == 0:
        return -math.inf
    with np.errstate(over='ignore'):
        barrier = float(np.divide(IONIZATION_ENERGY_EV, temperature))
    if math.isinf(barrier):
        return -math.inf  # Even log(k) cannot be represented at this input.
    tau = temperature / IONIZATION_ENERGY_EV
    answer = quad(lambda x: _scaled_bracket(x, tau) * math.exp(-x),
                  0., np.inf, epsabs=0., epsrel=_QUAD_RTOL, limit=200,
                  full_output=True)
    integral, error = answer[:2]
    if (len(answer) != 3 or not math.isfinite(integral) or integral <= 0
            or not math.isfinite(error) or error > 10 * _QUAD_RTOL * integral):
        message = answer[3] if len(answer) > 3 else 'Nonpositive/nonfinite or inaccurate quadrature.'
        raise RuntimeError(f'Ionization Maxwellian quadrature failed: {message}')
    return _LOG_PREFACTOR + .5 * math.log(temperature) - barrier + math.log(integral)


def log_k_ion(T_e_eV):
    """Natural log of the numerical rate in cm^3/s; same shape as input.

    k = sqrt(8*eV_to_erg/(pi*m_e))*T^(-3/2)
        * integral_I^inf E*sigma(E)*exp(-E/T) dE.

    With x=(E-I)/T, the activation exponential is evaluated outside the
    integral. This remains useful when k itself underflows in float64.
    log_k_ion(0)=-inf. No artificial T=1 eV cutoff is introduced.
    """
    temperature = _nonnegative_array(T_e_eV, 'T_e_eV')
    result = np.empty_like(temperature)
    for index in np.ndindex(temperature.shape):
        result[index] = _log_k_single(float(temperature[index]))
    return float(result) if result.ndim == 0 else result


def k_ion(T_e_eV):
    """<sigma_ion*v_e> [cm^3/s], T_e in eV; zero-drift Maxwellian.

    k(0)=0. Very low positive temperatures can give a float64 underflow to
    zero; use log_k_ion to distinguish this from an imposed physical cutoff.
    Intended for nonrelativistic temperatures; accepting an input does not
    establish accuracy outside the low-temperature plasma model.
    """
    with np.errstate(under='ignore'):
        result = np.exp(log_k_ion(T_e_eV))
    return float(result) if np.ndim(result) == 0 else result


def L_ion(T_e_eV):
    """Potential-energy loss coefficient I*k [erg cm^3/s].

    P_ion=n_e*n0*L_ion. No excitation radiation or extra 3T/2 or 5T cost is
    included. This is consistent with W excluding atomic binding energy.
    """
    return IONIZATION_ENERGY_ERG * k_ion(T_e_eV)


def make_ionization_coefficients(parameters, scale=0.0, n0_cm3=None):
    """Return (0,0) or theta -> (D_ion,E_ion) for one prescribed n0.

    D_ion=eta*L*n0*k_ion(T)/u_star, E_ion=(I/T_star_erg)*D_ion.
    The same eta multiplies particle creation and its energy cost. Source
    scales and neutral density are copied. Default off preserves old calls;
    it needs no source parameters and never evaluates the atomic integral.
    This first stage uses a constant prescribed n0 per trial, as existing CX.
    """
    scale = float(scale)
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('Ionization scale must be finite and between 0 and 1.')
    if scale == 0:
        return (0., 0.)
    length, speed, energy, temperature = (float(parameters[key]) for key in
        ('L_cm', 'U_star_cm_per_s', 'T_star_erg', 'T_star_eV'))
    n0 = float(parameters['n0_cm3'] if n0_cm3 is None else n0_cm3)
    if (not np.isfinite([length, speed, energy, temperature, n0]).all()
            or min(length, speed, energy, temperature) <= 0 or n0 < 0):
        raise ValueError('Need n0>=0 and positive finite L,u_star,T_star (CGS/eV).')
    if not np.isclose(energy, temperature*EV_TO_ERG, rtol=1e-10, atol=0):
        raise ValueError('T_star_erg and T_star_eV are inconsistent.')
    prefactor = scale * length * n0 / speed
    epsilon = IONIZATION_ENERGY_ERG / energy
    if not math.isfinite(prefactor) or not math.isfinite(epsilon):
        raise ValueError('Ionization prefactors are not representable.')
    if n0 == 0:
        return (0., 0.)

    def coefficients(theta):
        theta = float(theta)
        if not math.isfinite(theta) or theta < 0:
            raise ValueError('theta must be finite and nonnegative.')
        D_ion = prefactor * k_ion(theta*temperature)
        E_ion = epsilon * D_ion
        if not np.isfinite([D_ion, E_ion]).all():
            raise ValueError('Nonfinite ionization coefficients; check source scales.')
        return float(D_ion), float(E_ion)

    return coefficients


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[.1192, 1., 3.272, 14.936, 77.822])
    args = parser.parse_args()
    try:
        temperatures = _nonnegative_array(args.temperatures, 'temperatures')
        rates = k_ion(temperatures)
        logs = log_k_ion(temperatures)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print('Janev-Reiter-Samm (2003), H(1s) approximation for atomic D; no solver changes.')
    print('T_e [eV]       k_ion [cm^3/s]       L_ion [erg cm^3/s]       ln(k_ion)')
    for temperature, rate, logarithm in zip(temperatures, rates, logs):
        print(f'{temperature:12.6g}  {rate:18.9e}  '
              f'{IONIZATION_ENERGY_ERG * rate:22.9e}  {logarithm:14.6g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
