import csv
from pathlib import Path
import unittest


class UniverseManifestTests(unittest.TestCase):
    def test_issuer_ids_are_pinned_and_unique(self):
        path = Path(__file__).resolve().parents[1] / "configs/universe.csv"
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["ticker"] for row in rows}), 100)
        self.assertEqual(len({row["cik"] for row in rows}), 100)
        self.assertTrue(all(len(row["cik"]) == 10 and row["cik"].isdigit() for row in rows))
        self.assertEqual({sector: sum(r["sector"] == sector for r in rows)
                          for sector in {r["sector"] for r in rows}},
                         {"retail": 50, "manufacturing": 50})
