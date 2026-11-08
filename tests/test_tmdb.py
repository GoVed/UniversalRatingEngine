"""Unit tests for TMDb (The Movie Database) benchmark integration, season granularity, and consensus calibration."""

import json
import unittest
from pathlib import Path
import polars as pl

from src.build_unified_db import calculate_multisource_consensus


class TestTMDbIntegration(unittest.TestCase):
    def test_tmdb_data_artifacts_exist(self):
        """Verify that TMDb source artifacts exist and contain valid records."""
        tv_path = Path("data/tmdb/tmdb_tv_ratings.json")
        movie_path = Path("data/tmdb/tmdb_movie_ratings.parquet")
        wikidata_path = Path("data/tmdb/wikidata_imdb_tmdb_tv_map.json")

        self.assertTrue(tv_path.exists(), "tmdb_tv_ratings.json must exist")
        self.assertTrue(movie_path.exists(), "tmdb_movie_ratings.parquet must exist")
        self.assertTrue(wikidata_path.exists(), "wikidata_imdb_tmdb_tv_map.json must exist")

        with open(tv_path) as f:
            tv_data = json.load(f)
        self.assertGreater(len(tv_data), 1000, "Should have over 1,000 TMDb TV shows mapped")

        df_movies = pl.read_parquet(movie_path)
        self.assertGreater(len(df_movies), 200000, "Should have over 200,000 TMDb movie ratings")
        self.assertIn("tconst", df_movies.columns)
        self.assertIn("vote_average", df_movies.columns)
        self.assertIn("vote_count", df_movies.columns)

    def test_tmdb_tv_season_granularity(self):
        """Verify season-level rating granularity for serialized TV series."""
        with open("data/tmdb/tmdb_tv_ratings.json") as f:
            tv_data = json.load(f)

        # Breaking Bad (tt0903747)
        self.assertIn("tt0903747", tv_data)
        bb = tv_data["tt0903747"]
        self.assertIn("seasons", bb)
        self.assertIn("1", bb["seasons"])
        self.assertIn("5", bb["seasons"])
        # Season 5 should be rated higher than Season 1
        s1_vote = bb["seasons"]["1"]["vote_average"]
        s5_vote = bb["seasons"]["5"]["vote_average"]
        self.assertGreater(s5_vote, s1_vote)
        self.assertGreater(bb["vote_count"], 10000)

        # Game of Thrones (tt0944947)
        self.assertIn("tt0944947", tv_data)
        got = tv_data["tt0944947"]
        self.assertIn("seasons", got)
        self.assertIn("4", got["seasons"])
        self.assertIn("8", got["seasons"])
        # Season 4 should be rated significantly higher than Season 8
        self.assertGreater(got["seasons"]["4"]["vote_average"], got["seasons"]["8"]["vote_average"] + 1.5)

    def test_tmdb_in_unified_master_db(self):
        """Verify unified parquet master contains tmdb_raw and tmdb_votes."""
        parquet_path = Path("artifacts/combined/calibrated_scores.parquet")
        self.assertTrue(parquet_path.exists())
        df = pl.read_parquet(parquet_path)

        self.assertIn("tmdb_raw", df.columns)
        self.assertIn("tmdb_votes", df.columns)

        # Check TV seasons cohort has TMDb ratings
        tv_tmdb = df.filter((pl.col("media_type") == "tv_season") & pl.col("tmdb_raw").is_not_null())
        self.assertGreater(len(tv_tmdb), 1000, "Should have 1,000+ TV seasons with TMDb scores")

        # Check Cinema cohort has TMDb ratings
        movie_tmdb = df.filter((pl.col("media_type") == "movie") & pl.col("tmdb_raw").is_not_null())
        self.assertGreater(len(movie_tmdb), 10000, "Should have 10,000+ movies with TMDb scores")

        # Check Inception
        inception = df.filter(pl.col("tconst") == "tt1375666")
        self.assertEqual(len(inception), 1)
        self.assertIsNotNone(inception["tmdb_raw"][0])
        self.assertGreater(inception["tmdb_votes"][0], 20000)

    def test_tmdb_web_export_schema(self):
        """Verify web JSON export has tmdb_raw and tmdb_votes."""
        web_json = Path("docs/data/movies_combined.json")
        self.assertTrue(web_json.exists())
        with open(web_json) as f:
            d = json.load(f)

        cols = d["cols"]
        self.assertIn("tmdb_raw", cols)
        self.assertIn("tmdb_votes", cols)

        tmdb_r_idx = cols.index("tmdb_raw")
        tmdb_v_idx = cols.index("tmdb_votes")
        tc_idx = cols.index("tconst")
        title_idx = cols.index("title")

        # Verify Breaking Bad Season 5
        bb_s5 = [r for r in d["data"] if r[tc_idx] == "tt0903747" and "Season 5" in r[title_idx]]
        self.assertTrue(len(bb_s5) > 0)
        self.assertIsNotNone(bb_s5[0][tmdb_r_idx])
        self.assertGreaterEqual(bb_s5[0][tmdb_r_idx], 8.5)
        self.assertGreater(bb_s5[0][tmdb_v_idx], 10000)

    def test_tmdb_consensus_calibration(self):
        """Verify TMDb consensus math and calibration weighting."""
        # Baseline with IMDb alone
        mod_b_imdb, _, _ = calculate_multisource_consensus(
            imdb_raw=8.0,
            imdb_votes=50000,
        )

        # Adding a higher TMDb score should pull mod_b upward
        mod_b_with_tmdb, _, _ = calculate_multisource_consensus(
            imdb_raw=8.0,
            imdb_votes=50000,
            tmdb_raw=8.8,
            tmdb_votes=30000,
        )
        self.assertGreater(mod_b_with_tmdb, mod_b_imdb)

        # Negligible TMDb votes should have negligible pull
        mod_b_negligible, _, _ = calculate_multisource_consensus(
            imdb_raw=8.0,
            imdb_votes=50000,
            tmdb_raw=9.5,
            tmdb_votes=10,
        )
        self.assertAlmostEqual(mod_b_negligible, mod_b_imdb, delta=0.05)


if __name__ == "__main__":
    unittest.main()
