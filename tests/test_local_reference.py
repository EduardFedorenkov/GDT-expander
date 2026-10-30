"""The local reference is immutable evidence, not a source of solver guesses."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from compare_local_reference import (
    DEFAULT_REFERENCE, ROOT_FILE, GRID_FILE, read_table, load_run, compare_runs,
)


class LocalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.copy = Path(self.temp.name)/'candidate'
        shutil.copytree(DEFAULT_REFERENCE, self.copy)

    def edit_table(self, name, modify):
        rows = read_table(self.copy/name)
        fields = tuple(rows[0])
        modify(rows)
        with (self.copy/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter=';')
            writer.writeheader()
            writer.writerows(rows)

    def test_reference_has_all_36_pairs_and_64_roots(self):
        manifest, grid, roots = load_run(DEFAULT_REFERENCE)
        self.assertEqual(manifest['lambdas'], [7., 8., 9., 10.])
        self.assertEqual(len(grid), 36)
        self.assertEqual(sum(map(len, roots.values())), 64)
        for (lam, n0), values in roots.items():
            self.assertEqual(len(values), 0 if n0 == 1e12 else 2)

    def test_reference_compares_to_itself_exactly(self):
        result = compare_runs(DEFAULT_REFERENCE, DEFAULT_REFERENCE)
        self.assertTrue(result['passed'])
        self.assertEqual(result['matched_roots'], 64)
        self.assertTrue(all(v == 0 for v in result['max_absolute_differences'].values()))
        self.assertEqual(result['diagnostic_changes'], [])

    def test_local_ids_paths_and_row_order_do_not_identify_roots(self):
        def edit(rows):
            rows.reverse()
            for i, row in enumerate(rows):
                row['solution_id'] = f'changed_{i}'
                row['local_solution_id'] = f'local_{i}'
                row['case_report'] = 'different/output/path'
        self.edit_table(ROOT_FILE, edit)
        self.assertTrue(compare_runs(DEFAULT_REFERENCE, self.copy)['passed'])

    def test_changed_temperature_fails_comparison(self):
        self.edit_table(ROOT_FILE,
            lambda rows: rows[0].update(T_eV=str(float(rows[0]['T_eV'])*1.001)))
        self.assertFalse(compare_runs(DEFAULT_REFERENCE, self.copy)['passed'])

    def test_missing_pair_and_inconsistent_root_count_are_errors(self):
        self.edit_table(GRID_FILE, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Incomplete/extra'):
            compare_runs(DEFAULT_REFERENCE, self.copy)
        shutil.copy2(DEFAULT_REFERENCE/GRID_FILE, self.copy/GRID_FILE)
        self.edit_table(ROOT_FILE, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'root count mismatch'):
            compare_runs(DEFAULT_REFERENCE, self.copy)

    def test_changed_physics_is_not_accepted_as_a_refactor(self):
        path = self.copy/'manifest.json'
        data = json.loads(path.read_text())
        data['ionization_scale'] = 0.
        path.write_text(json.dumps(data), encoding='utf-8')
        self.assertFalse(compare_runs(DEFAULT_REFERENCE, self.copy)['passed'])

    def test_counter_changes_are_reported_separately(self):
        self.edit_table(GRID_FILE, lambda rows: rows[0].update(n_starts='1'))
        result = compare_runs(DEFAULT_REFERENCE, self.copy)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['diagnostic_changes']), 1)

    def test_nonfinite_values_and_bad_tolerance_are_not_silently_accepted(self):
        self.edit_table(ROOT_FILE, lambda rows: rows[0].update(T_eV='nan'))
        with self.assertRaises(ValueError):
            compare_runs(DEFAULT_REFERENCE, self.copy)
        with self.assertRaises(ValueError):
            compare_runs(DEFAULT_REFERENCE, DEFAULT_REFERENCE, rtol=0)


if __name__ == '__main__':
    unittest.main()
