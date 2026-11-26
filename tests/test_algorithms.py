import unittest

import numpy as np

from adaptive_low_rank.algorithms import Adaptive, BatchMax, Greedy, GreedyPP, Random


class ScaleInvarianceTests(unittest.TestCase):
    def test_selection_is_unchanged_under_rescaling(self):
        matrix = np.random.default_rng(42).standard_normal((5, 7))
        for algorithm in (Adaptive, BatchMax, Greedy, GreedyPP, Random):
            with self.subTest(algorithm=algorithm.__name__):
                normal = algorithm().select_columns(
                    matrix, k=3, random_state=4, n_candidates=3
                )
                scaled = algorithm().select_columns(
                    1e-5 * matrix, k=3, random_state=4, n_candidates=3
                )
                np.testing.assert_array_equal(normal.indices, scaled.indices)
                np.testing.assert_allclose(
                    normal.residuals * 1e-5, scaled.residuals, rtol=1e-10
                )

    def test_scaled_full_rank_matrix(self):
        for algorithm in (Adaptive, BatchMax, Greedy, GreedyPP, Random):
            with self.subTest(algorithm=algorithm.__name__):
                result = algorithm().select_columns(
                    1e-5 * np.eye(2), k=2, random_state=0, n_candidates=2
                )
                self.assertEqual(len(set(result.indices)), 2)
                self.assertLess(result.residuals[-1], 1e-15)

    def test_scaled_batch_max_gains(self):
        result = BatchMax().select_columns(
            1e-5 * np.eye(2), k=1, random_state=0, n_candidates=2, compute_alpha=True
        )
        self.assertTrue(np.isfinite(result.gains_bm[0]))
        self.assertTrue(np.isfinite(result.gains_as[0]))
        self.assertGreater(result.gains_as[0], 0)

    def test_zero_matrix_fails_explicitly(self):
        for algorithm in (Adaptive, BatchMax, Greedy, GreedyPP):
            with self.subTest(algorithm=algorithm.__name__):
                with self.assertRaises(ValueError):
                    algorithm().select_columns(
                        np.zeros((2, 2)), k=1, random_state=0, n_candidates=2
                    )

    def test_exhausted_rank_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "residual energy"):
            Greedy().select_columns(np.ones((2, 2)), k=2)

    def test_greedy_pp_picks_best_uniformly_sampled_candidate(self):
        matrix = np.array([[1e-5, 0.0, 0.0, 4.0], [0.0, 1.0, 1.0, 0.0]])
        for seed in (0, 5):
            with self.subTest(seed=seed):
                candidates = np.random.RandomState(seed).choice(4, size=3, replace=False)
                residuals = []
                for idx in candidates:
                    column = matrix[:, idx]
                    projected = matrix - np.outer(column, column @ matrix) / (
                        column @ column
                    )
                    residuals.append(np.linalg.norm(projected, ord="fro"))
                idx, _ = GreedyPP().select_index(
                    matrix, k=1, random_state=np.random.RandomState(seed), n_candidates=3
                )
                self.assertEqual(idx, candidates[np.argmin(residuals)])

    def test_greedy_pp_fails_when_sampled_candidate_is_exhausted(self):
        matrix = np.eye(2)
        first = GreedyPP().select_columns(matrix, k=1, random_state=1, n_candidates=1)
        np.testing.assert_array_equal(first.indices, [0])
        with self.assertRaisesRegex(ValueError, "negligible residual energy"):
            GreedyPP().select_columns(matrix, k=2, random_state=1, n_candidates=1)

    def test_random_samples_indices_independently_of_column_energy(self):
        matrix = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        expected = np.random.RandomState(0).choice(3, size=3, replace=False)
        result = Random().select_columns(matrix, k=3, random_state=0)
        zero_result = Random().select_columns(np.zeros_like(matrix), k=3, random_state=0)
        np.testing.assert_array_equal(result.indices, expected)
        np.testing.assert_array_equal(zero_result.indices, expected)
        np.testing.assert_allclose(zero_result.residuals, 0.0)
        np.testing.assert_allclose(result.residuals, [np.sqrt(2), 0.0, 0.0], atol=1e-15)

    def test_random_rejects_more_selections_than_columns(self):
        with self.assertRaisesRegex(ValueError, "number of columns"):
            Random().select_columns(np.eye(2), k=3, random_state=0)

    def test_random_residuals_match_least_squares(self):
        matrix = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        result = Random().select_columns(matrix, k=3, random_state=0)
        for count, residual in enumerate(result.residuals, 1):
            selected = matrix[:, result.indices[:count]]
            coefficients = np.linalg.lstsq(selected, matrix, rcond=None)[0]
            expected = np.linalg.norm(matrix - selected @ coefficients, ord="fro")
            self.assertAlmostEqual(residual, expected, places=12)

    def test_random_retains_near_dependent_selections(self):
        matrix = np.array([[0.0, 1.0, 1.0], [1.0, 1e-17, 0.0]])
        for scale in (1.0, 1e-5):
            with self.subTest(scale=scale):
                result = Random().select_columns(scale * matrix, k=3, random_state=0)
                np.testing.assert_array_equal(result.indices, [2, 1, 0])
                np.testing.assert_allclose(
                    result.residuals, scale * np.array([1.0, 1.0, 0.0]), atol=1e-20
                )


if __name__ == "__main__":
    unittest.main()
