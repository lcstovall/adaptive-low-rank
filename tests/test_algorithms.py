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
        for algorithm in (Adaptive, BatchMax, Greedy, GreedyPP, Random):
            with self.subTest(algorithm=algorithm.__name__):
                with self.assertRaises(ValueError):
                    algorithm().select_columns(
                        np.zeros((2, 2)), k=1, random_state=0, n_candidates=2
                    )

    def test_exhausted_rank_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "residual energy"):
            Greedy().select_columns(np.ones((2, 2)), k=2)


if __name__ == "__main__":
    unittest.main()
