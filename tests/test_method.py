import unittest

import numpy as np

from har.model import DIM, VECTORS, fit_metric, rebuild
from har.native import (
    CHECKPOINT_WORDS, SNAPSHOT_WORDS, CONFLICT, DUPLICATE, ERASED, FULL, INVALID, IO, OK, STALE,
    Edit, Journal,
    c_embeddings, features, pack_blocks,
)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1702)
        self.x = self.rng.integers(-128, 128, size=(12, VECTORS, DIM), dtype=np.int16)
        self.labels = np.repeat(np.arange(6, dtype=np.uint8), 2)
        self.packed = pack_blocks(self.x, self.labels)
        self.edit = Edit(1, 0, 2, 0, 1)

    def compare(self, journal, labels):
        sums, counts = rebuild(self.x, labels)
        np.testing.assert_array_equal(journal.stats()[0], sums)
        np.testing.assert_array_equal(journal.stats()[1], counts)
        np.testing.assert_array_equal(journal.labels(), labels)

    def test_duplicate_conflict_and_reboot(self):
        journal = Journal(self.packed)
        self.assertEqual(journal.apply(self.edit), (OK, 6))
        journal.reboot()
        before = journal.words.copy()
        self.assertEqual(journal.apply(self.edit), (DUPLICATE, 0))
        self.assertEqual(journal.apply(Edit(1, 0, 2, 0, 4)), (CONFLICT, 0))
        self.assertEqual(journal.apply(Edit(2, 0, 3, 0, 4)), (INVALID, 0))
        self.assertEqual(journal.apply(Edit(0, 0, 1, 1, 0)), (INVALID, 0))
        np.testing.assert_array_equal(journal.words, before)

    def test_all_edit_boundaries_and_seeded_torn_masks(self):
        moved = self.labels.copy()
        moved[:2] = 1
        masks = [0, 0x0000FFFF, 0xFFFF0000, ERASED]
        masks.extend(int(x) for x in self.rng.integers(0, 2**32, size=36, dtype=np.uint64))
        for cut in range(6):
            for mask in masks:
                with self.subTest(cut=cut, mask=mask):
                    journal = Journal(self.packed)
                    self.assertEqual(journal.apply(self.edit, cut=cut, mask=mask)[0], IO)
                    journal.reboot()
                    durable = bool(journal.state.last_id)
                    self.compare(journal, moved if durable else self.labels)
                    self.assertEqual(journal.apply(self.edit)[0], DUPLICATE if durable else OK)
                    self.compare(journal, moved)

    def test_full_journal_compacts_and_uses_other_bank(self):
        journal = Journal(self.packed, slots=1)
        self.assertEqual(journal.apply(self.edit), (OK, 6))
        reverse = Edit(2, 0, 2, 1, 0)
        self.assertEqual(journal.apply(reverse), (FULL, 0))
        self.assertEqual(journal.compact(), (OK, 1 + CHECKPOINT_WORDS))
        journal.reboot()
        self.assertEqual(journal.state.bank, 1)
        self.assertEqual(journal.apply(self.edit), (DUPLICATE, 0))
        self.assertEqual(journal.apply(Edit(1, 0, 2, 0, 3)), (CONFLICT, 0))
        self.assertEqual(journal.apply(reverse), (OK, 6))
        self.assertEqual(journal.compact(), (OK, 1 + CHECKPOINT_WORDS))
        journal.reboot()
        self.assertEqual((journal.state.bank, journal.state.generation), (0, 2))
        self.assertEqual(journal.apply(self.edit), (STALE, 0))
        self.assertEqual(journal.apply(reverse), (DUPLICATE, 0))
        self.compare(journal, self.labels)

    def test_all_checkpoint_boundaries_and_inactive_bank_erase(self):
        journal = Journal(self.packed, slots=4)
        journal.apply(self.edit)
        journal.compact()
        journal.apply(Edit(2, 0, 2, 1, 0))
        journal.compact()
        for cut in range(CHECKPOINT_WORDS + 1):
            for mask in (0, 0x5555AAAA, ERASED):
                candidate = Journal(self.packed, slots=4, words=journal.words.copy())
                self.assertEqual(candidate.compact(cut=cut, mask=mask)[0], IO)
                candidate.reboot()
                self.compare(candidate, self.labels)
                durable = cut == CHECKPOINT_WORDS and mask == ERASED
                self.assertEqual(candidate.state.generation, 2 + durable)
                if not durable:
                    self.assertEqual(candidate.compact()[0], OK)
                candidate.reboot()
                self.assertEqual(candidate.state.generation, 3)
                self.assertEqual(candidate.apply(Edit(2, 0, 2, 1, 0)), (DUPLICATE, 0))
                self.compare(candidate, self.labels)

    def test_snapshot_cost_recovery_and_stale_acknowledgement(self):
        moved = self.labels.copy()
        moved[:2] = 1
        for cut in range(SNAPSHOT_WORDS + 1):
            for mask in (0, 0x0000FFFF, ERASED):
                journal = Journal(self.packed)
                self.assertEqual(journal.apply(self.edit, cut=cut, mask=mask, snapshot=True)[0], IO)
                journal.reboot()
                durable = cut == SNAPSHOT_WORDS and mask == ERASED
                self.compare(journal, moved if durable else self.labels)
                status = (STALE, 0) if durable else (OK, SNAPSHOT_WORDS + 1)
                self.assertEqual(journal.apply(self.edit, snapshot=True), status)
                journal.reboot()
                self.compare(journal, moved)
        self.assertEqual(journal.compact(), (OK, CHECKPOINT_WORDS + 1))
        journal.reboot()
        self.compare(journal, moved)

    def test_summary_vectors_and_bit_identical_normalization(self):
        for layout in ("summary", "vectors", "packed"):
            journal = Journal(pack_blocks(self.x, self.labels, layout))
            self.assertEqual(journal.apply(self.edit), (OK, 6))
            journal.reboot()
            labels = self.labels.copy()
            labels[:2] = 1
            self.compare(journal, labels)
        raw = features(self.rng.integers(-2000, 2000, size=(9, 100, 6), dtype=np.int16))
        metric = fit_metric(raw)
        np.testing.assert_array_equal(metric.apply(raw), c_embeddings(raw, metric))

    def test_packed_boundaries_and_word_crossings(self):
        x = self.rng.integers(-2047, 2048, size=self.x.shape, dtype=np.int16)
        x[0] = np.tile([2047, -2047, 0, -1], DIM // 4)
        x[1] = np.tile([-2047, 2047, -1, 1], DIM // 4)
        self.x = x
        journal = Journal(pack_blocks(x, self.labels, "packed"))
        self.compare(journal, self.labels)
        self.assertEqual(journal.apply(self.edit), (OK, 6))
        self.assertEqual(journal.compact()[0], OK)
        journal.reboot()
        expected = self.labels.copy()
        expected[:2] = 1
        self.compare(journal, expected)
        self.assertEqual(journal.apply(self.edit), (DUPLICATE, 0))

    def test_checkpoint_receipt_ablation(self):
        for keep in (True, False):
            journal = Journal(self.packed, checkpoint_receipt=keep)
            self.assertEqual(journal.apply(self.edit), (OK, 6))
            journal.compact()
            expected = DUPLICATE if keep else STALE
            self.assertEqual(journal.apply(self.edit), (expected, 0))
            journal.reboot()
            self.assertEqual(journal.apply(self.edit), (expected, 0))

    def test_snapshot_with_matched_receipt(self):
        journal = Journal(self.packed, snapshot_receipt=True)
        self.assertEqual(journal.apply(self.edit, snapshot=True), (OK, 1 + CHECKPOINT_WORDS))
        journal.reboot()
        self.assertEqual(journal.apply(self.edit, snapshot=True), (DUPLICATE, 0))
        self.assertEqual(journal.apply(Edit(1, 0, 2, 0, 3), snapshot=True), (CONFLICT, 0))

    def test_edit_horizon_and_epoch(self):
        limited = Journal(self.packed, edit_floor=2)
        self.assertEqual(limited.apply(self.edit), (INVALID, 0))
        journal = Journal(self.packed, epoch=14)
        journal.apply(self.edit)
        journal.compact()
        newer = Journal(self.packed, epoch=15, words=journal.words.copy())
        self.compare(newer, self.labels)
        self.assertEqual(newer.state.last_id, 0)


if __name__ == "__main__":
    unittest.main()
