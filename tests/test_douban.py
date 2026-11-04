"""Unit tests for Douban benchmark pipeline and calibration."""

import unittest
from pathlib import Path
import polars as pl

from src.build_douban_db import parse_douban_genres, extract_year


class TestDoubanPipeline(unittest.TestCase):
    def test_parse_douban_genres(self):
        self.assertEqual(parse_douban_genres("剧情/动作/武侠"), "Drama|Action|Wuxia")
        self.assertEqual(parse_douban_genres("喜剧/爱情"), "Comedy|Romance")
        self.assertEqual(parse_douban_genres("科幻/冒险/动画"), "Sci-Fi|Adventure|Animation")
        self.assertEqual(parse_douban_genres(""), "Drama")
        self.assertEqual(parse_douban_genres(None), "Drama")

    def test_extract_year(self):
        self.assertEqual(extract_year("1993-01-01(中国香港)"), 1993)
        self.assertEqual(extract_year("2021-08-15"), 2021)
        self.assertEqual(extract_year("2000"), 2000)
        self.assertEqual(extract_year(""), 0)
        self.assertEqual(extract_year(None), 0)

    def test_douban_artifacts_integrity(self):
        cal_path = Path("artifacts/douban/calibrated_scores.parquet")
        self.assertTrue(cal_path.exists(), "Douban calibrated parquet artifact should exist")
        df = pl.read_parquet(cal_path)
        self.assertGreater(len(df), 30000, "Douban should contain over 30,000 calibrated titles")

        # Verify key mathematical properties
        expected_cols = [
            "movieId", "title", "year", "genres", "vote_count",
            "raw_avg", "bayesian_avg", "score_model_a", "score_model_b",
            "score_model_c", "bias_b_i", "raw_rank", "calibrated_rank",
            "rank_delta", "calibration_delta", "rank_diagnostic"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check ratings are bounded
        self.assertGreaterEqual(float(df["raw_avg"].min()), 1.0)
        self.assertLessEqual(float(df["raw_avg"].max()), 10.0)
        self.assertGreaterEqual(float(df["score_model_b"].min()), 1.0)
        self.assertLessEqual(float(df["score_model_b"].max()), 10.0)

    def test_douban_web_assets(self):
        json_path = Path("docs/data/movies_douban.json")
        self.assertTrue(json_path.exists(), "movies_douban.json should exist")
        import json
        with open(json_path) as f:
            data = json.load(f)
        self.assertEqual(data["dataset"], "douban")
        self.assertGreater(data["total_titles"], 30000)
        self.assertIn("tags", data["cols"])


if __name__ == "__main__":
    unittest.main()
