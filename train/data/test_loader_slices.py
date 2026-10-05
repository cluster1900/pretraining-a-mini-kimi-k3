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

    def test_stream_exhaustion_is_loud_and_never_wraps(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shard.bin"
            with path.open("wb") as handle:
                array("I", range(4)).tofile(handle)
            stream = SourceStream("s", [(path, 0, 4)])
            self.assertEqual(stream.take(4).tolist(), [0, 1, 2, 3])
            with self.assertRaises(EOFError):
                stream.take(1)
            stream.close()

    def test_missing_mix_source_is_rejected(self):
        loader = MultiSourceDataLoader()
        with self.assertRaises(KeyError):
            loader.set_mix({"missing-source": 1.0})



def _write_manifest(root: Path, sources: dict) -> Path:
    """sources: name -> (weight, token values)."""
    import json
    entries = {}
    for name, (weight, values) in sources.items():
        shard = root / f"{name}.bin"
        with shard.open("wb") as handle:
            array("I", values).tofile(handle)
        entries[name] = {
            "weight": weight, "shards": [str(shard)],
            "shard_metadata": [{"path": str(shard), "tokens": len(values), "bytes": 4 * len(values)}],
        }
    path = root / "manifest.json"
    path.write_text(json.dumps({"sources": entries}))
    return path


class LoaderTests(unittest.TestCase):
    def test_batches_are_int64_and_exact(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            big = 4_000_000_000  # > int32 max, valid uint32
            manifest = _write_manifest(Path(td), {"a": (1.0, [big, 1, 2, 3, 4, 5, 6, 7])})
            loader = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=2)
            self.addCleanup(loader.close)
            x, y = loader.next_batch("cpu")
            self.assertEqual(x.dtype, torch.int64)
            self.assertEqual(x.tolist(), [[big, 1, 2, 3], [4, 5, 6, 7]])
            self.assertTrue(torch.equal(x, y))

    def test_exhausted_positive_weight_source_raises(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_manifest(Path(td), {
                "a": (0.5, list(range(4))),          # one sequence only
                "b": (0.5, list(range(100))),
            })
            loader = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            self.addCleanup(loader.close)
            loader.next_batch("cpu")  # a
            loader.next_batch("cpu")  # b
            with self.assertRaisesRegex(EOFError, "'a'.*exhausted"):
                loader.next_batch("cpu")  # a again: must not fall back to b

    def test_zero_weight_source_is_never_sampled(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_manifest(Path(td), {
                "a": (1.0, list(range(40))), "z": (0.0, list(range(40))),
            })
            loader = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            self.addCleanup(loader.close)
            for _ in range(10):
                loader.next_batch("cpu")
            self.assertEqual(loader.token_counts["z"], 0)

    def test_set_mix_is_noop_when_unchanged_and_state_restores_mix(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_manifest(Path(td), {
                "a": (0.5, list(range(400))), "b": (0.5, list(range(400))),
            })
            decay = {"a": 0.25, "b": 0.75}
            loader = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            self.addCleanup(loader.close)
            self.assertFalse(loader.set_mix({"a": 0.5, "b": 0.5}))
            self.assertTrue(loader.set_mix(decay))
            for _ in range(5):
                loader.next_batch("cpu")
            served = dict(loader.mix_tokens_served)
            self.assertFalse(loader.set_mix(decay))  # keeps the deficit counters
            self.assertEqual(loader.mix_tokens_served, served)
            state = loader.state_dict()
            expected = [loader.next_batch("cpu")[0].tolist() for _ in range(6)]

            # Resume: a fresh loader in the stable mix, then the saved state.
            resumed = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            self.addCleanup(resumed.close)
            resumed.set_mix(decay)
            resumed.load_state_dict(state)
            self.assertTrue(resumed.same_mix(decay))
            self.assertEqual([resumed.next_batch("cpu")[0].tolist() for _ in range(6)], expected)

            # Even without set_mix the saved target mix is restored.
            plain = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            self.addCleanup(plain.close)
            plain.load_state_dict(state)
            self.assertTrue(plain.same_mix(decay))
            self.assertEqual([plain.next_batch("cpu")[0].tolist() for _ in range(6)], expected)

    def test_snapshot_is_not_mutated_and_replays_same_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = _write_manifest(Path(td), {
                "a": (0.5, list(range(400))), "b": (0.5, list(range(1000, 1400))),
            })
            loader = MultiSourceDataLoader(str(manifest), seq_len=4, batch_size=1)
            self.addCleanup(loader.close)
            snapshot = loader.state_dict()
            frozen = repr(snapshot)
            first = [loader.next_batch("cpu")[0].tolist() for _ in range(5)]
            for _ in range(3):
                loader.load_state_dict(snapshot)
                self.assertEqual([loader.next_batch("cpu")[0].tolist() for _ in range(5)], first)
            self.assertEqual(repr(snapshot), frozen)


if __name__ == "__main__":
    unittest.main()
