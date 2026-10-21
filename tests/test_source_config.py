"""Source normalization, throat flows and config compatibility checks."""

import json
from math import isfinite, pi, sqrt
from pathlib import Path
import unittest
from unittest.mock import patch

from simple_cx_model import load_cx_config, throat_fluxes
from simple_cx_model.constants import EV_TO_ERG, M_D_G


def source_config():
    return dict(
        A_m_cm2=pi * 3.5**2, n_c_cm3=1e13, T_c_eV=100.0, chi_E=8.0,
        k_cx_cm3_per_s=1e-8, lambda_sh=5.0, L_cm=184.5, n0_cm3=1e14,
    )


def load_values(values):
    with patch("pathlib.Path.read_text", return_value=json.dumps(values)):
        return load_cx_config("in_memory.json")


class ThroatFluxTests(unittest.TestCase):
    def test_physical_density_normalization_and_units(self):
        area, density, temperature = pi * 3.5**2, 5e12, 100.0
        flows = throat_fluxes(area, density, temperature)
        self.assertAlmostEqual(flows["J0_per_s"] / 1.0627842743810317e21, 1)
        self.assertAlmostEqual(flows["M0_dyn"] / 61658.98270331338, 1)
        self.assertAlmostEqual(flows["W0_erg_per_s"] / 1.362214505116747e12, 1)
        self.assertAlmostEqual(
            flows["J0_per_s"] / (area * density),
            sqrt(2 * temperature * EV_TO_ERG / (pi * M_D_G)),
        )

    def test_chi_changes_only_energy(self):
        first = throat_fluxes(40, 5e12, 100, 8)
        second = throat_fluxes(40, 5e12, 100, 10)
        self.assertEqual(first["J0_per_s"], second["J0_per_s"])
        self.assertEqual(first["M0_dyn"], second["M0_dyn"])
        self.assertAlmostEqual(second["W0_erg_per_s"] / first["W0_erg_per_s"], 1.25)

    def test_custom_mass_is_used(self):
        first = throat_fluxes(40, 5e12, 100)
        second = throat_fluxes(40, 5e12, 100, m_D_g=2 * M_D_G)
        self.assertAlmostEqual(second["J0_per_s"] / first["J0_per_s"], 1 / sqrt(2))
        self.assertEqual(first["M0_dyn"], second["M0_dyn"])
        self.assertAlmostEqual(second["W0_erg_per_s"] / first["W0_erg_per_s"], 1 / sqrt(2))

    def test_invalid_inputs(self):
        valid = dict(A_m=40, n_m=5e12, T_m=100, chi_E=8, m_D_g=M_D_G)
        for name in valid:
            for bad in (0, -1, float("nan"), float("inf"), "bad", None):
                with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                    throat_fluxes(**dict(valid, **{name: bad}))

    def test_unrepresentable_flows_are_rejected(self):
        for values in ((1e308, 1e308, 100), (40, 5e12, 1e-320)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                throat_fluxes(*values)


class SourceConfigTests(unittest.TestCase):
    def test_source_conversion_and_legacy_output_keys(self):
        config = source_config()
        p = load_values(config)
        self.assertEqual(p["n_m_cm3"], config["n_c_cm3"] / 2)
        self.assertEqual(p["T_m_eV"], config["T_c_eV"])
        self.assertEqual(p["A0_cm2"], config["A_m_cm2"])
        self.assertEqual(p["n0_cm3"], config["n0_cm3"])
        self.assertEqual(p["m_D_g"], M_D_G)
        self.assertAlmostEqual(p["T_star_eV"], config["chi_E"] * config["T_c_eV"])
        self.assertAlmostEqual(p["beta"], config["chi_E"] / (2 * pi))
        for name in ("J0_per_s", "M0_dyn", "W0_erg_per_s", "A_star_cm2",
                     "U_star_cm_per_s", "n_star_cm3", "D"):
            self.assertTrue(isfinite(p[name]) and p[name] > 0, name)
        expected = throat_fluxes(config["A_m_cm2"], config["n_c_cm3"] / 2,
                                 config["T_c_eV"], config["chi_E"])
        for name in expected:
            self.assertEqual(p[name], expected[name])

    def test_config_mass_propagates_to_fluxes_and_scales(self):
        config = dict(source_config(), m_D_g=2 * M_D_G)
        p = load_values(config)
        expected = throat_fluxes(config["A_m_cm2"], config["n_c_cm3"] / 2,
                                 config["T_c_eV"], config["chi_E"],
                                 m_D_g=config["m_D_g"])
        self.assertEqual(p["J0_per_s"], expected["J0_per_s"])
        self.assertAlmostEqual(p["beta"], config["chi_E"] / (2 * pi))

    def test_original_central_density_formulas(self):
        config = source_config()
        p = load_values(config)
        area, density = config["A_m_cm2"], config["n_c_cm3"]
        temperature = config["T_c_eV"] * EV_TO_ERG
        self.assertAlmostEqual(
            p["J0_per_s"] / (area * density * sqrt(temperature / (2 * pi * M_D_G))), 1)
        self.assertAlmostEqual(p["M0_dyn"] / (area * density * temperature), 1)

    def test_chi_rescales_derived_energy_not_particle_flow(self):
        first = load_values(source_config())
        second = load_values(dict(source_config(), chi_E=10))
        self.assertEqual(first["J0_per_s"], second["J0_per_s"])
        self.assertEqual(first["M0_dyn"], second["M0_dyn"])
        self.assertAlmostEqual(second["beta"] / first["beta"], 1.25)

    def test_sample_json_uses_new_format(self):
        path = Path(__file__).resolve().parents[1] / "simple_cx_model" / "cx_config.json"
        p = load_cx_config(path)
        self.assertEqual(p["n_m_cm3"], p["n_c_cm3"] / 2)
        self.assertEqual(p["A0_cm2"], p["A_m_cm2"])

    def test_derived_input_overrides_are_rejected(self):
        for name in ("J0_per_s", "M0_dyn", "W0_erg_per_s", "A0_cm2"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                load_values(dict(source_config(), **{name: 1}))

    def test_missing_source_inputs(self):
        for name in ("A_m_cm2", "n_c_cm3", "T_c_eV", "chi_E"):
            config = source_config()
            del config[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                load_values(config)

    def test_invalid_numeric_inputs(self):
        for name in source_config():
            for bad in (0, -1, float("nan"), float("inf"), "bad", None):
                with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                    load_values(dict(source_config(), **{name: bad}))

    def test_json_root_must_be_an_object(self):
        for value in ([], None, 1, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_values(value)


if __name__ == "__main__":
    unittest.main()
