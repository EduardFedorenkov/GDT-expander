"""Console source/CGS summary; no physical solver change is required."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from isothermal_cx_model.scan_lambda import main, print_source_summary


def source_parameters():
    # Deliberately independent precomputed flows: printing must not derive them.
    return dict(n_c_cm3=3e13, T_c_eV=150.0, n_m_cm3=1.5e13,
                A_m_cm2=38.48451, chi_E=8.0,
                J0_per_s=1.23456789e21, M0_dyn=4.56789123e5,
                W0_erg_per_s=7.89123456e12)


class SourceSummaryTests(unittest.TestCase):
    def test_prints_source_and_precomputed_flows_with_units(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_source_summary(source_parameters())
        text = output.getvalue()
        for expected in (
            "n_c = 3.000000e+13 cm^-3", "T_c = 150 eV",
            "n_m = 1.500000e+13 cm^-3", "A_m = 38.4845 cm^2", "chi_E = 8",
            "J_0 = 1.234568e+21 particles/s",
            "M_0 = 4.567891e+05 dyn (g*cm/s^2)",
            "W_0 = 7.891235e+12 erg/s",
        ):
            self.assertIn(expected, text)

    def test_parameters_are_not_modified(self):
        parameters = source_parameters()
        original = copy.deepcopy(parameters)
        with contextlib.redirect_stdout(io.StringIO()):
            print_source_summary(parameters)
        self.assertEqual(parameters, original)

    def test_main_prints_once_before_scan_even_when_no_roots(self):
        output = io.StringIO()

        def scan(*args, **kwargs):
            self.assertEqual(output.getvalue().count("J_0 ="), 1)
            print("SCAN_STARTED")
            return [], []

        with tempfile.TemporaryDirectory() as folder:
            argv = ["scan_lambda", "--lambdas", "5", "6", "--densities", "1e14",
                    "--output", str(Path(folder)/"results"), "--no-plot"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(output), \
                    patch("isothermal_cx_model.scan_lambda.load_cx_config",
                          return_value=source_parameters()), \
                    patch("isothermal_cx_model.scan_lambda.scan_lambda_neutral",
                          side_effect=scan) as scanner:
                main()
        self.assertEqual(scanner.call_count, 1)
        self.assertEqual(output.getvalue().count("J_0 ="), 1)
        self.assertLess(output.getvalue().index("W_0 ="),
                        output.getvalue().index("SCAN_STARTED"))


if __name__ == "__main__":
    unittest.main()
