"""Unit and mathematical tests for Model A, Model B, and Model C."""

import unittest
import numpy as np
import polars as pl
import torch

from src.models.z_score import UserNormalizedZScore
from src.models.matrix_factorization import BiasedMatrixFactorization
from src.models.bridge_calibration import BridgeCalibrationModel


class TestModels(unittest.TestCase):
    def setUp(self):
        # Create a controlled synthetic dataset:
        # User 1: Harsh critic (ratings: 4, 5, 6 -> mean 5.0, std 1.0)
        # User 2: Generous fan (ratings: 8, 9, 10 -> mean 9.0, std 1.0)
        # Movie 101: rated 6 by User 1 (z = +1.0), rated 10 by User 2 (z = +1.0)
        # Movie 102: rated 4 by User 1 (z = -1.0), rated 8 by User 2 (z = -1.0)
        # Movie 103: rated 5 by User 1 (z = 0.0), rated 9 by User 2 (z = 0.0)
        self.ratings = pl.DataFrame({
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [101, 102, 103, 101, 102, 103],
            "rating": [6.0, 4.0, 5.0, 10.0, 8.0, 9.0],
            "timestamp": [1000, 1001, 1002, 1003, 1004, 1005],
        })

        self.movies = pl.DataFrame({
            "movieId": [101, 102, 103],
            "title": ["Masterpiece (2020)", "Mediocre (2021)", "Average (2022)"],
            "genres": ["Drama|Sci-Fi", "Comedy", "Drama"],
        })

    def test_model_a_user_zscore(self):
        model = UserNormalizedZScore(eps=1e-6)
        model.fit(self.ratings)
        scores_df = model.get_item_scores()

        # Both users gave their highest rating (+1 z-score) to Movie 101
        m101 = scores_df.filter(pl.col("movieId") == 101)
        self.assertAlmostEqual(m101["mean_z"][0], 1.0, places=3)

        # Both users gave their lowest rating (-1 z-score) to Movie 102
        m102 = scores_df.filter(pl.col("movieId") == 102)
        self.assertAlmostEqual(m102["mean_z"][0], -1.0, places=3)

        # Movie 103 is at the mean (0 z-score)
        m103 = scores_df.filter(pl.col("movieId") == 103)
        self.assertAlmostEqual(m103["mean_z"][0], 0.0, places=3)

        # Calibrated scores must maintain the ordering
        self.assertGreater(m101["score_model_a"][0], m103["score_model_a"][0])
        self.assertGreater(m103["score_model_a"][0], m102["score_model_a"][0])

        # Prediction works
        pred = model.predict(user_id=1, movie_id=101)
        self.assertAlmostEqual(pred, 6.0, places=2)

    def test_model_b_biased_mf(self):
        # Expand synthetic dataset to give enough interactions for MF
        np.random.seed(42)
        num_users = 20
        num_items = 10
        records = []
        # Item 0 is high quality (+2.0 bias), Item 9 is low quality (-2.0 bias)
        for u in range(num_users):
            u_bias = np.random.normal(0, 0.5)
            for i in range(num_items):
                i_bias = 2.0 if i == 0 else (-2.0 if i == 9 else 0.0)
                r = np.clip(7.0 + u_bias + i_bias + np.random.normal(0, 0.2), 1.0, 10.0)
                records.append({"userId": u + 1, "movieId": i + 1, "rating": float(r)})

        df = pl.DataFrame(records)
        model = BiasedMatrixFactorization(
            k=8,
            lambda_reg=0.01,
            learning_rate=0.05,
            epochs=15,
            batch_size=64,
            device="cpu",
        )
        model.fit(df, verbose=False)

        scores_df = model.get_item_scores()
        self.assertEqual(len(scores_df), num_items)

        # Item 1 (index 0) must have higher latent bias than Item 10 (index 9)
        score_high = scores_df.filter(pl.col("movieId") == 1)["score_model_b"][0]
        score_low = scores_df.filter(pl.col("movieId") == 10)["score_model_b"][0]
        self.assertGreater(score_high, score_low)

        # Predictions fall cleanly within bounds
        pred = model.predict(user_id=1, movie_id=1)
        self.assertGreaterEqual(pred, 1.0)
        self.assertLessEqual(pred, 10.0)

    def test_model_c_shannon_entropy_and_bridge_calibration(self):
        # User 1: watches only Drama (10 ratings of Drama) -> entropy should be 0.0
        # User 2: watches Drama, Comedy, Sci-Fi, Horror equally -> high entropy
        ratings_c = pl.DataFrame({
            "userId": [1]*4 + [2]*4,
            "movieId": [201, 202, 203, 204, 201, 205, 206, 207],
            "rating": [8.0, 8.5, 9.0, 8.5, 7.0, 7.5, 6.5, 7.0],
        })

        movies_c = pl.DataFrame({
            "movieId": [201, 202, 203, 204, 205, 206, 207],
            "title": [f"Film {i}" for i in range(1, 8)],
            "genres": ["Drama", "Drama", "Drama", "Drama", "Comedy", "Sci-Fi", "Horror"],
        })

        model = BridgeCalibrationModel(
            bridge_percentile=0.50,
            loyalist_genre_threshold=0.50,
            shrinkage_weight=1.0,
        )
        model.fit(ratings_c, movies_c, verbose=False)

        entropy_df = model.user_entropy_df
        u1_ent = entropy_df.filter(pl.col("userId") == 1)["diversity_index"][0]
        u2_ent = entropy_df.filter(pl.col("userId") == 2)["diversity_index"][0]

        # User 1 has only 1 genre, so entropy is 0
        self.assertAlmostEqual(u1_ent, 0.0, places=3)
        # User 2 has multiple genres, so diversity index > 0
        self.assertGreater(u2_ent, 0.5)

        # Loyalist vs Bridge: User 1 is Drama loyalist (rated Drama at ~8.5)
        # User 2 is Bridge user (rated Drama at 7.0)
        # Therefore, Drama multiplier should be < 1.0 (deflating enthusiast inflation)
        g_mults = model.get_genre_multipliers()
        drama_row = g_mults.filter(pl.col("genre") == "Drama")
        self.assertLess(drama_row["inflation_multiplier"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
