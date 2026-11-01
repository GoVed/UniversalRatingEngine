"""Unit tests for data loader and cold-start filtering."""

import unittest
import polars as pl

from src.data_loader import filter_cold_start, load_raw_data


class TestDataLoader(unittest.TestCase):
    def test_filter_cold_start_iterative(self):
        # Create user rating counts:
        # User 1: 20 ratings
        # User 2: 5 ratings (below min 15)
        # User 3: 20 ratings
        # Item 101: 35 ratings
        # Item 102: 10 ratings (below min 30)
        
        # Construct synthetic ratings
        u1_ratings = [{"userId": 1, "movieId": 101, "rating": 4.0, "timestamp": 100} for _ in range(20)]
        u2_ratings = [{"userId": 2, "movieId": 101, "rating": 3.0, "timestamp": 101} for _ in range(5)]
        u3_ratings = [{"userId": 3, "movieId": 101, "rating": 5.0, "timestamp": 102} for _ in range(10)] + \
                     [{"userId": 3, "movieId": 102, "rating": 2.0, "timestamp": 103} for _ in range(10)]
        
        all_ratings = u1_ratings + u2_ratings + u3_ratings
        ratings_df = pl.DataFrame(all_ratings)

        movies_df = pl.DataFrame({
            "movieId": [101, 102],
            "title": ["Movie 101", "Movie 102"],
            "genres": ["Action", "Comedy"],
        })

        filtered_ratings, filtered_movies = filter_cold_start(
            ratings_df,
            movies_df,
            min_user_ratings=15,
            min_item_ratings=30,
        )

        # User 2 had only 5 ratings, so excluded
        self.assertNotIn(2, filtered_ratings["userId"].to_list())

        # Movie 102 had only 10 ratings, so excluded
        self.assertNotIn(102, filtered_ratings["movieId"].to_list())
        self.assertNotIn(102, filtered_movies["movieId"].to_list())

        # All surviving users have >= 15 ratings
        user_counts = filtered_ratings["userId"].value_counts()["count"].to_list()
        for count in user_counts:
            self.assertGreaterEqual(count, 15)

        # All surviving items have >= 30 ratings
        item_counts = filtered_ratings["movieId"].value_counts()["count"].to_list()
        for count in item_counts:
            self.assertGreaterEqual(count, 30)

    def test_load_raw_data_deduplication(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Write a ratings.csv with duplicates for user 1 on movie 10
            ratings_csv = tmp_path / "ratings.csv"
            ratings_csv.write_text(
                "userId,movieId,rating,timestamp\n"
                "1,10,3.0,1000\n"
                "1,10,4.5,2000\n"  # Newer duplicate rating
                "2,10,4.0,1500\n"
            )
            
            movies_csv = tmp_path / "movies.csv"
            movies_csv.write_text(
                "movieId,title,genres\n"
                "10,Test Movie,Action\n"
            )

            ratings_df, movies_df = load_raw_data(tmp_path, scale_to_10=True)

            # Total rows should be 2 (deduplicated), keeping latest (4.5 * 2 = 9.0)
            self.assertEqual(len(ratings_df), 2)
            u1_row = ratings_df.filter(pl.col("userId") == 1)
            self.assertEqual(len(u1_row), 1)
            self.assertEqual(u1_row["rating"][0], 9.0)
            self.assertEqual(u1_row["timestamp"][0], 2000)

    def test_get_combined_data_namespacing_and_dedup(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.data_loader import get_combined_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dir1 = tmp_path / "cohort1"
            dir2 = tmp_path / "cohort2"
            dir1.mkdir()
            dir2.mkdir()

            # Cohort 1: User 1 rates Movie 100
            (dir1 / "ratings.csv").write_text(
                "userId,movieId,rating,timestamp\n" +
                "\n".join([f"1,100,4.0,{1000 + i}" for i in range(16)])
            )
            (dir1 / "movies.csv").write_text("movieId,title,genres\n100,Title 100,Drama\n")

            # Cohort 2: User 1 rates Movie 100 (same userId, should be namespaced!)
            (dir2 / "ratings.csv").write_text(
                "userId,movieId,rating,timestamp\n" +
                "\n".join([f"1,100,5.0,{2000 + i}" for i in range(16)])
            )
            (dir2 / "movies.csv").write_text("movieId,title,genres\n100,Title 100,Drama\n")

            def mock_download(src, data_dir="data"):
                return dir1 if src == "cohort1" else dir2

            with patch("src.data_loader.download_and_extract", side_effect=mock_download):
                r_comb, m_comb = get_combined_data(
                    sources=["cohort1", "cohort2"],
                    data_dir=tmp_path,
                    min_user_ratings=1,
                    min_item_ratings=1,
                )

                # Cohort 1 user is 1; Cohort 2 user is 1 + 1_000_000_000
                unique_users = sorted(r_comb["userId"].unique().to_list())
                self.assertEqual(unique_users, [1, 1_000_000_001])
                self.assertEqual(m_comb["movieId"].to_list(), [100])

    def test_mal_download_and_extract_format(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.data_loader import download_and_extract

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_mal = tmp_path / "mal"
            target_mal.mkdir()
            # Fake raw_mal.csv
            (target_mal / "raw_mal.csv").write_text(
                "user_id,anime_id,rating,genres,name,average_rating,anime_rating,image url\n"
                "10,500,9,\"Action, Sci-Fi\",Test Anime,8.5,PG-13,http://img\n"
            )

            with patch("src.data_loader._download_file"):
                out_dir = download_and_extract("mal", data_dir=tmp_path)
                self.assertTrue((out_dir / "ratings.csv").exists())
                self.assertTrue((out_dir / "movies.csv").exists())

                m_df = pl.read_csv(out_dir / "movies.csv")
                self.assertEqual(m_df["title"][0], "Test Anime")
                self.assertEqual(m_df["genres"][0], "Action|Sci-Fi")

                r_df = pl.read_csv(out_dir / "ratings.csv")
                self.assertEqual(r_df["userId"][0], 10)
                self.assertEqual(r_df["movieId"][0], 500)
                self.assertEqual(r_df["rating"][0], 9.0)

    def test_mal_57m_download_and_extract_format(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.data_loader import download_and_extract

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_mal = tmp_path / "mal-57m"
            target_mal.mkdir()
            # Fake anime.csv (Hernan Valdivieso schema)
            (target_mal / "anime.csv").write_text(
                "MAL_ID,Name,Score,Genres,English name\n"
                "1,Cowboy Bebop,8.78,\"Action, Adventure, Comedy, Sci-Fi\",Cowboy Bebop\n"
            )
            # Fake rating_complete.csv (profoz schema)
            (target_mal / "rating_complete.csv").write_text(
                "user_id,anime_id,rating\n"
                "100,1,10\n"
                "101,1,9\n"
            )

            with patch("src.data_loader._download_file"):
                out_dir = download_and_extract("mal-57m", data_dir=tmp_path)
                self.assertTrue((out_dir / "ratings.csv").exists())
                self.assertTrue((out_dir / "movies.csv").exists())

                m_df = pl.read_csv(out_dir / "movies.csv")
                self.assertEqual(m_df["movieId"][0], 1)
                self.assertEqual(m_df["title"][0], "Cowboy Bebop")
                self.assertEqual(m_df["genres"][0], "Action|Adventure|Comedy|Sci-Fi")

                r_df = pl.read_csv(out_dir / "ratings.csv")
                self.assertEqual(len(r_df), 2)
                self.assertEqual(r_df["userId"][0], 100)
                self.assertEqual(r_df["movieId"][0], 1)
                self.assertEqual(r_df["rating"][0], 10.0)
                self.assertEqual(r_df["timestamp"][0], 0)

    def test_mal_7m_download_and_extract_format(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.data_loader import download_and_extract

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_mal = tmp_path / "mal-7m"
            target_mal.mkdir()
            # Fake raw_anime.csv (CooperUnion schema)
            (target_mal / "raw_anime.csv").write_text(
                "anime_id,name,genre,type,episodes,rating,members\n"
                "1535,Death Note,\"Mystery, Police, Psychological, Supernatural, Thriller\",TV,37,8.71,1013917\n"
            )
            # Fake raw_rating.csv (CooperUnion schema, with -1 unrated)
            (target_mal / "raw_rating.csv").write_text(
                "user_id,anime_id,rating\n"
                "50,1535,10\n"
                "51,1535,-1\n"
            )

            with patch("src.data_loader._download_file"):
                out_dir = download_and_extract("mal-7m", data_dir=tmp_path)
                self.assertTrue((out_dir / "ratings.csv").exists())
                self.assertTrue((out_dir / "movies.csv").exists())

                m_df = pl.read_csv(out_dir / "movies.csv")
                self.assertEqual(m_df["movieId"][0], 1535)
                self.assertEqual(m_df["title"][0], "Death Note")
                self.assertEqual(m_df["genres"][0], "Mystery|Police|Psychological|Supernatural|Thriller")

                r_df = pl.read_csv(out_dir / "ratings.csv")
                self.assertEqual(len(r_df), 1)  # -1 filtered out
                self.assertEqual(r_df["userId"][0], 50)
                self.assertEqual(r_df["movieId"][0], 1535)
                self.assertEqual(r_df["rating"][0], 10.0)


if __name__ == "__main__":
    unittest.main()
