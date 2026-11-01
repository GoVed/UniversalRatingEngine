"""Integration tests for CLI interface."""

import unittest
from typer.testing import CliRunner

from run import app

runner = CliRunner()


class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("download", result.output)
        self.assertIn("fit", result.output)
        self.assertIn("inspect", result.output)
        self.assertIn("report", result.output)
        self.assertIn("top", result.output)
        self.assertIn("compare", result.output)

    def test_inspect_existing_title(self):
        # Inspect on pre-trained ML-1M dataset
        result = runner.invoke(app, ["inspect", "Princess Mononoke", "--dataset", "ml-1m"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Princess Mononoke", result.output)
        self.assertIn("Model A", result.output)
        self.assertIn("Model B", result.output)
        self.assertIn("Raw Arithmetic Average", result.output)
        self.assertIn("Overall Rank Shift", result.output)

    def test_top_command(self):
        result = runner.invoke(app, ["top", "--dataset", "ml-1m", "-n", "5"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Top 5 Titles", result.output)

    def test_compare_command(self):
        result = runner.invoke(app, ["compare", "Toy Story", "Heat", "--dataset", "ml-1m"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Head-to-Head", result.output)
        self.assertIn("Toy", result.output)
        self.assertIn("Heat", result.output)

    def test_imdb_standalone_top_and_inspect(self):
        result = runner.invoke(app, ["top", "--dataset", "imdb", "-n", "3"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Top 3 Titles", result.output)

        result_inspect = runner.invoke(app, ["inspect", "The Shawshank Redemption", "--dataset", "imdb"])
        self.assertEqual(result_inspect.exit_code, 0)
        self.assertIn("The Shawshank Redemption", result_inspect.output)
        self.assertIn("Model B", result_inspect.output)


if __name__ == "__main__":
    unittest.main()
