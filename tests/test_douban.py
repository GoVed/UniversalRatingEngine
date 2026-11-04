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

    def test_douban_in_unified_master_db(self):
        cal_path = Path("artifacts/combined/calibrated_scores.parquet")
        self.assertTrue(cal_path.exists(), "Combined calibrated parquet artifact should exist")
        df = pl.read_parquet(cal_path)
        self.assertIn("douban_raw", df.columns)
        self.assertIn("douban_votes", df.columns)
        self.assertGreater(len(df), 60000, "Unified DB should have 60,000+ titles with Douban included")

        douban_entries = df.filter(pl.col("douban_raw").is_not_null())
        self.assertGreater(len(douban_entries), 20000, "Over 20,000 titles should have Douban scores in Unified DB")

        json_path = Path("docs/data/movies_combined.json")
        self.assertTrue(json_path.exists(), "movies_combined.json should exist")
        import json
        with open(json_path) as f:
            data = json.load(f)
        self.assertEqual(data["dataset"], "combined")
        self.assertIn("douban_raw", data["cols"])
        self.assertIn("douban_votes", data["cols"])
        self.assertGreater(data["total_titles"], 60000)


if __name__ == "__main__":
    unittest.main()
