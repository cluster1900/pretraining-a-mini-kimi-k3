"""Per-rank token slices cover a shard once and keep every rank supplied."""
import unittest
from array import array
from pathlib import Path
import tempfile
from loader import MultiSourceDataLoader, SourceStream, rank_token_range, segments_for_rank


class SliceTests(unittest.TestCase):
    def test_ranges_partition_every_shard(self):
        for n_tokens in (0, 1, 3, 50_000_000, 39_726_521):
            for world in (1, 4):
                covered = []
                for rank in range(world):
                    start, end = rank_token_range(n_tokens, rank, world)
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(end, n_tokens)
                    covered.append((start, end))
                cursor = 0
                for start, end in covered:
                    self.assertEqual(start, cursor)
                    cursor = end
                self.assertEqual(cursor, n_tokens)

    def test_four_ranks_each_clear_the_tight_sources(self):
        # Shard sizes from the audited v2 manifest. Whole-shard striding left
        # two ranks below the 10B mixture; slices must not.
        need = {"chinese": 356_250_255, "code": 306_250_220}
        shards = {
            "chinese": [50_000_000] * 28 + [39_726_521],
            "code": [50_000_000] * 25 + [38_200_254],
        }
        for name, sizes in shards.items():
            items = [(Path(f"s{i}"), n) for i, n in enumerate(sizes)]
            totals = []
            for rank in range(4):
                segments = segments_for_rank(items, rank, 4)
                totals.append(sum(end - start for _, start, end in segments))
            self.assertEqual(sum(totals), sum(sizes))
            self.assertGreaterEqual(min(totals), need[name])

    def test_slices_read_each_token_once(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shard.bin"
            values = array("I", range(10))
            with path.open("wb") as handle:
                values.tofile(handle)
            seen = []
            for rank in range(4):
                start, end = rank_token_range(10, rank, 4)
                stream = SourceStream("s", [(path, start, end)])
                seen.extend(stream.take(end - start).tolist())
                stream.close()
            self.assertEqual(seen, values.tolist())

    def test_missing_mix_source_is_rejected(self):
        loader = MultiSourceDataLoader()
        with self.assertRaises(KeyError):
            loader.set_mix({"missing-source": 1.0})


if __name__ == "__main__":
    unittest.main()
