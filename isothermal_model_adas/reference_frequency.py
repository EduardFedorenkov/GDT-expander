"""Simple per-profile search scale: max(c_s/(L-a), CX, ion, rec, MAR, MAD, MAI).

Reaction frequencies are rates per plasma ion, evaluated on a SAVED profile.
No source derivatives, cooling estimate or high-frequency proof is implied.
The 10*nu_ref ceiling is a configurable search heuristic, not a spectral bound.
Run this module to print frequencies only, without any spectral integration.
"""

import argparse
import json
from math import isfinite
import sys

import numpy as np

from .linearized_model import saved_cases


def reference_frequency(parameters, state, factor=10.):
    """Return frequencies in s^-1 and the proposed real ceiling for one root.

    D_X = (L/U_star)*n_neutral*k_X, R=(L/U_star)*n_star*alpha_rec.
    Thus nu_X=D_X/t_star; nu_rec=R*N/t_star, N=n_e/n_star.
    The existing providers have already split n_D and n_D2: do NOT multiply
    by molecular_fraction a second time. Excitation is not in this simple rule.
    """
    factor = float(factor)
    if not isfinite(factor) or factor <= 0:
        raise ValueError("Frequency multiplier must be finite and positive.")
    if state.get("status") != "integrated":
        raise ValueError("A successfully reconstructed stationary profile is required.")
    g, N = (np.asarray(state[key],dtype=float) for key in ("g","n_over_n_star"))
    if (g.ndim != 1 or g.size < 2 or N.shape != g.shape
            or not np.isfinite(g).all() or not np.isfinite(N).all()
            or np.any(N <= 0) or np.any(np.diff(g) <= 0)):
        raise ValueError("Profile coordinates and positive electron densities must be finite and aligned.")
    L, U, beta = (float(parameters[key]) for key in ("L_cm","U_star_cm_per_s","beta"))
    alpha, theta = float(state["alpha"]),float(state["theta"])
    if (not all(isfinite(v) and v > 0 for v in (L,U,beta,theta))
            or not isfinite(alpha) or not 0 <= alpha < 1
            or not np.isclose(g[0],alpha,rtol=0,atol=1e-10)
            or not np.isclose(g[-1],1.,rtol=0,atol=1e-10)):
        raise ValueError("Invalid scales or profile does not cover [alpha,1].")
    time_star = L/U
    c_s = U*np.sqrt(2*beta*theta)
    rates = {}

    def add(name, values):
        rate = np.broadcast_to(np.asarray(values,dtype=float),N.shape)
        if not np.isfinite(rate).all() or np.any(rate < 0):
            raise ValueError(f"{name}: frequencies must be finite and nonnegative.")
        index = int(np.argmax(rate))
        rates[name] = dict(max_per_s=float(rate[index]),min_per_s=float(np.min(rate)),
                           z_at_max_cm=float(L*g[index]))

    add("acoustic",c_s/(L*(1-alpha)))
    rates["acoustic"]["z_at_max_cm"] = None  # a layer-wide scale, not a local reaction
    add("cx",np.asarray(state["D"])/time_star)
    add("ionization",np.asarray(state["D_ion_profile"])/time_star)
    add("recombination",np.asarray(state["R_profile"])*N/time_star)
    molecular = float(parameters.get("molecular_fraction",0.)) > 0
    for branch in ("mar","mad","mai"):
        key = f"D_{branch}_profile"
        if molecular and key not in state:
            raise ValueError(f"Active molecules but missing saved profile coefficient: {key}")
        add(branch,np.asarray(state.get(key,0.))/time_star)
    dominant = max(rates,key=lambda name: rates[name]["max_per_s"])
    nu_ref = rates[dominant]["max_per_s"]
    ceiling = factor*nu_ref
    if not isfinite(ceiling) or ceiling <= 0:
        raise ValueError("The frequency-derived ceiling is not finite and positive.")
    return dict(nu_ref_per_s=nu_ref,dominant_process=dominant,
                factor=factor,s_real_max_per_s=ceiling,
                c_s_cm_per_s=float(c_s),layer_length_cm=float(L*(1-alpha)),
                tau_acoustic_s=float(L*(1-alpha)/c_s),
                rates=rates,profile_points=int(g.size),
                heuristic_not_proven_spectral_bound=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan",required=True)
    parser.add_argument("--n0",type=float,required=True,help="Neutral density, cm^-3.")
    parser.add_argument("--lambda-sh",type=float,required=True)
    parser.add_argument("--root-index",type=int,action="append")
    parser.add_argument("--factor",type=float,default=10.)
    args = parser.parse_args(argv)
    if not isfinite(args.factor) or args.factor <= 0:
        parser.error("--factor must be finite and positive.")
    if not all(isfinite(v) and v > 0 for v in (args.n0,args.lambda_sh)):
        parser.error("n0 and lambda must be finite and positive.")
    try:
        roots = saved_cases(args.scan,args.n0,args.lambda_sh)
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))
    if args.root_index is not None:
        roots = [bg for bg in roots if bg.root_index in args.root_index]
    report = dict(n0_cm3=args.n0,lambda_sh=args.lambda_sh,roots=[])
    for bg in roots:
        row = dict(root_index=bg.root_index,T_eV=bg.seed["T_eV"],a_cm=bg.alpha*bg.p["L_cm"])
        try:
            # No variation of theta, j_t or alpha and no stationary root search.
            state = bg.trial()
            row["reference_frequency"] = reference_frequency(bg.p,state,args.factor)
            row["status"] = "frequency_computed_spectrum_not_searched"
        except (ValueError,RuntimeError,FloatingPointError) as error:
            row.update(status="unresolved",error=str(error))
        report["roots"].append(row)
        print(f"root {bg.root_index}: {row['status']}",file=sys.stderr,flush=True)
    if not roots:
        report["status"] = "no_saved_verified_roots"
    print(json.dumps(report,indent=2,allow_nan=False))
    return int(not roots or any(row["status"] == "unresolved" for row in report["roots"]))


if __name__ == "__main__":
    raise SystemExit(main())
