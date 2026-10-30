"""Profile shape labels are diagnostic, not physical branch identifiers."""

import unittest
from isothermal_cx_model.diagnostics import classify_profile


class DiagnosticTests(unittest.TestCase):
    def test_monotone_and_constant(self):
        for values in ([1, .8, .5], [1, 2, 3], [1, 1, 1]):
            self.assertEqual(classify_profile(values), "monotone")

    def test_maximum(self):
        self.assertEqual(classify_profile([1, 1.2, .5]), "with_maximum")

    def test_nonmonotone(self):
        self.assertEqual(classify_profile([1, 1.2, .8, 1.1, .5]), "nonmonotone")


if __name__ == "__main__":
    unittest.main()
