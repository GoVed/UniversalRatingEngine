"""Unit tests for analytical metrics and ranking calculations."""

import unittest
import numpy as np
import polars as pl

from src.metrics import (
    calculate_metrics_summary,
    compute_bayesian_average,
    compute_cross_genre_summary,
    compute_rank_correlation,
    format_rank_change,
)


class TestMetrics(unittest.TestCase):
    def test_bayesian_average_scalar(self):
        # 10 votes at 9.0, global mean 7.0, threshold 50
        # WR = (10/60)*9.0 + (50/60)*7.0 = 1.5 + 5.833 = 7.333
        wr = compute_bayesian_average(raw_avg=9.0, vote_count=10, global_mean=7.0, min_votes_threshold=50)
        self.assertAlmostEqual(wr, (10 / 60) * 9.0 + (50 / 60) * 7.0, places=4)

        # High votes: 5000 votes at 9.0 -> should be very close to 9.0
        wr_high = compute_bayesian_average(raw_avg=9.0, vote_count=5000, global_mean=7.0, min_votes_threshold=50)
        self.assertGreater(wr_high, 8.95)

    def test_bayesian_average_polars_expr(self):
        df = pl.DataFrame({
            "raw_avg": [9.0, 5.0],
            "vote_count": [10, 100],
        }).with_columns(
            compute_bayesian_average(pl.col("raw_avg"), pl.col("vote_count"), global_mean=7.0, min_votes_threshold=50).alias("wr")
        )
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df["wr"][0], (10 / 60) * 9.0 + (50 / 60) * 7.0, places=4)

    def test_format_rank_change(self):
        # Dropped: raw rank 100, cal rank 512 -> +412
        msg_drop = format_rank_change(100, 512)
        self.assertIn("Dropped 412 ranks due to enthusiast bias", msg_drop)

        # Gained: raw rank 500, cal rank 415 -> -85
        msg_gain = format_rank_change(500, 415)
        self.assertIn("Gained 85 ranks due to harsh audience penalty", msg_gain)

        # Unchanged
        msg_same = format_rank_change(50, 50)
        self.assertIn("Rank unchanged", msg_same)

    def test_rmse_and_mae(self):
        y_true = np.array([7.0, 8.0, 9.0])
        y_pred = np.array([6.0, 8.0, 11.0])
        res = calculate_metrics_summary(y_true, y_pred)
        # diffs: [-1, 0, 2] -> squares [1, 0, 4] -> mean 5/3 -> rmse sqrt(5/3)
        self.assertAlmostEqual(res["rmse"], np.sqrt(5 / 3), places=4)
        # abs diffs: [1, 0, 2] -> mean 1.0
        self.assertAlmostEqual(res["mae"], 1.0, places=4)

    def test_rank_correlation(self):
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        corr = compute_rank_correlation(s1, s2)
        self.assertAlmostEqual(corr, 1.0, places=4)

    def test_title_article_normalization(self):
        from src.engine import format_display_title
        from src.export_web_data import clean_movie_title

        # Test engine formatter (preserves year if present)
        self.assertEqual(format_display_title("Shawshank Redemption, The (1994)"), "The Shawshank Redemption (1994)")
        self.assertEqual(format_display_title("Sixth Sense, The (1999)"), "The Sixth Sense (1999)")
        self.assertEqual(format_display_title("Dark Knight, The"), "The Dark Knight")
        self.assertEqual(format_display_title("Usual Suspects, The"), "The Usual Suspects")
        self.assertEqual(format_display_title("Great Escape, A (1963)"), "A Great Escape (1963)")
        self.assertEqual(format_display_title("Affair to Remember, An (1957)"), "An Affair to Remember (1957)")
        self.assertEqual(format_display_title("Fabuleux Destin d'Amélie Poulain, Le"), "Le Fabuleux Destin d'Amélie Poulain")

        # Test web cleaner (strips year and uninverts article)
        self.assertEqual(clean_movie_title("Shawshank Redemption, The (1994)"), "The Shawshank Redemption")
        self.assertEqual(clean_movie_title("Wizard of Oz, The (1939)"), "The Wizard of Oz")
        self.assertEqual(clean_movie_title("Schindler's List (1993)"), "Schindler's List")


if __name__ == "__main__":
    unittest.main()
