"""Paired per-participant experiments and reset campaigns against independent oracles."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, DEFAULT_DATA, LABELS, NOISE_PAIRS, prepare
from .model import (
    CLASSES, DIM, PRIOR_COUNT, VECTORS, centroids, f1_score, fit_metric, predict, rebuild, statistics,
)
from .native import (
    CHECKPOINT_WORDS, DUPLICATE, FULL, IO, OK,
    Edit, Journal, ROOT, SLOT_WORDS, c_predict, pack_blocks,
)

SEEDS = (17, 43, 101)
LENGTHS = (1, 2, 3)
BLOCKS_PER_CLASS = 4
RESULTS = ROOT / "results"
MODES = (
    "source_only", "no_correction", "add_only", "no_prior", "short_history",
    "coarse_blocks", "full_vectors", "unpacked_summary", "proposed",
)
FAULT_PATTERNS = {
    "before_word": 0,
    "torn_low_half": 0x0000FFFF,
    "torn_high_half": 0xFFFF0000,
    "word_complete": 0xFFFFFFFF,
}


def calibration_indices(data, subject, rng):
    candidates = np.flatnonzero((data["subject"] == subject) & ~data["test"])
    chosen = []
    for label in range(CLASSES):
        rows = candidates[data["label"][candidates] == label]
        groups = np.unique(np.column_stack((data["trial"][rows], data["group"][rows])), axis=0)
        runs = []
        for trial, group in groups:
            run = rows[(data["trial"][rows] == trial) & (data["group"][rows] == group)]
            if len(run) >= BLOCKS_PER_CLASS:
                runs.append(run)
        if not runs:
            raise ValueError(f"Participant {subject}, class {label} has no suitable calibration interval")
        run = runs[rng.integers(len(runs))]
        offset = rng.integers(len(run) - BLOCKS_PER_CLASS + 1)
        chosen.extend(run[offset:offset + BLOCKS_PER_CLASS])
    return np.sort(np.asarray(chosen))


def make_edits(labels, dataset, rng, length):
    edits = []
    corrupted = labels.copy()
    # Choose three classes per participant. The seeds vary the affected classes and subintervals.
    for label in sorted(rng.choice(CLASSES, size=3, replace=False)):
        indices = np.flatnonzero(labels == label)
        offset = int(rng.integers(len(indices) - length + 1))
        first = int(indices[offset])
        end = first + length
        wrong = NOISE_PAIRS[dataset][label]
        if not np.all(labels[first:end] == label):
            raise AssertionError("Calibration edit crossed an activity boundary")
        corrupted[first:end] = wrong
        edits.append(Edit(len(edits) + 1, first, end, wrong, int(label)))
    return corrupted, edits


def assert_model(journal, blocks, labels, prior, prior_count=PRIOR_COUNT):
    sums, counts = rebuild(blocks, labels)
    np.testing.assert_array_equal(journal.labels(), labels)
    np.testing.assert_array_equal(journal.stats()[0], sums)
    np.testing.assert_array_equal(journal.stats()[1], counts)
    model = centroids(sums, counts, prior, prior_count)
    np.testing.assert_array_equal(journal.prototypes(prior, prior_count), model)
    return model


def apply_edits(journal, edits):
    for edit in edits:
        if journal.apply(edit) != (OK, SLOT_WORDS):
            raise AssertionError(f"Rejected expected edit: {edit}")


def evaluate_case(dataset, subject, seed, length, blocks, labels, x, y, prior):
    rng = np.random.default_rng([seed, subject, length])
    corrupted, edits = make_edits(labels, dataset, rng, length)
    summary = pack_blocks(blocks, corrupted, "packed")
    proposal = Journal(summary)
    apply_edits(proposal, edits)
    proposed = assert_model(proposal, blocks, labels, prior)
    proposal.reboot()
    assert_model(proposal, blocks, labels, prior)
    unpacked = Journal(pack_blocks(blocks, corrupted))
    apply_edits(unpacked, edits)
    unpacked.reboot()
    unpacked_summary = assert_model(unpacked, blocks, labels, prior)

    vector_reference = Journal(pack_blocks(blocks, corrupted, "vectors"))
    apply_edits(vector_reference, edits)
    vector_reference.reboot()
    full_vectors = assert_model(vector_reference, blocks, labels, prior)

    sums, counts = rebuild(blocks, corrupted)
    add_sums, add_counts = sums.copy(), counts.copy()
    for edit in edits:
        fixed = blocks[edit.first:edit.end]
        add_sums[edit.to_label] += fixed.sum(axis=(0, 1))
        add_counts[edit.to_label] += len(fixed) * VECTORS

    recent = Journal(summary, edit_floor=len(blocks) - 8)
    for edit in edits:
        if edit.first >= recent.view.edit_floor:
            if recent.apply(edit) != (OK, SLOT_WORDS):
                raise AssertionError("Recent edit failed")

    # A two-second implementation only accepts an edit aligned to complete pairs of blocks.
    coarse = Journal(summary)
    for edit in edits:
        if edit.first % 2 == 0 and edit.end % 2 == 0:
            if coarse.apply(edit) != (OK, SLOT_WORDS):
                raise AssertionError("Aligned edit failed")

    models = {
        "source_only": prior,
        "no_correction": centroids(sums, counts, prior, PRIOR_COUNT),
        "add_only": centroids(add_sums, add_counts, prior, PRIOR_COUNT),
        "no_prior": proposal.prototypes(prior, 0),
        "short_history": recent.prototypes(prior, PRIOR_COUNT),
        "coarse_blocks": coarse.prototypes(prior, PRIOR_COUNT),
        "full_vectors": full_vectors,
        "unpacked_summary": unpacked_summary,
        "proposed": proposed,
    }
    target_prediction = c_predict(x, proposed)
    np.testing.assert_array_equal(target_prediction, predict(x, proposed))
    main_rows, class_rows, conf_rows = [], [], []
    details = {"dataset": dataset, "seed": seed, "subject": subject, "edit_length_s": length}
    for mode, model in models.items():
        predictions = c_predict(x, model)
        macro, by_class, matrix = f1_score(y, predictions)
        main_rows.append({
            **details, "mode": mode, "macro_f1": macro, "accuracy": float(np.mean(y == predictions)),
            "prediction_agreement": float(np.mean(target_prediction == predictions)),
            "test_windows": len(y),
        })
        if length == 2 and mode in {"no_correction", "add_only", "proposed"}:
            for i, class_name in enumerate(LABELS[dataset]):
                class_rows.append({**details, "mode": mode, "class": class_name,
                                   "f1": by_class[i], "windows": int(matrix[i].sum())})
            if mode in {"no_correction", "proposed"}:
                for i, actual in enumerate(LABELS[dataset]):
                    for j, predicted in enumerate(LABELS[dataset]):
                        conf_rows.append({**details, "mode": mode, "actual": actual,
                                          "predicted": predicted, "count": int(matrix[i, j])})
    return main_rows, class_rows, conf_rows, corrupted, edits


def is_model(journal, wanted):
    sums, counts, labels = wanted
    got_sums, got_counts = journal.stats()
    return (np.array_equal(got_sums, sums) and np.array_equal(got_counts, counts)
            and np.array_equal(journal.labels(), labels))


def fault_campaign(dataset, subject, seed, blocks, corrupted, edits):
    packed = pack_blocks(blocks, corrupted, "packed")
    first, second, third = edits
    prefix = Journal(packed)
    apply_edits(prefix, [first])
    pre = corrupted.copy()
    pre[first.first:first.end] = first.to_label
    post = pre.copy()
    post[second.first:second.end] = second.to_label
    pre_state = (*rebuild(blocks, pre), pre)
    post_state = (*rebuild(blocks, post), post)
    rows = []
    context = {"dataset": dataset, "subject": subject, "seed": seed}
    slot = prefix.slot_offset(prefix.state.next_slot)
    complete = Journal(packed, words=prefix.words.copy())
    apply_edits(complete, [second])
    expected = complete.words[slot:slot + SLOT_WORDS]
    for fault, mask in FAULT_PATTERNS.items():
        for index in range(SLOT_WORDS + 1):
            if index == SLOT_WORDS and fault != "before_word":
                continue
            journal = Journal(packed, words=prefix.words.copy())
            status, writes = journal.apply(second, cut=index, mask=mask)
            if status != (OK if index == SLOT_WORDS else IO):
                raise AssertionError("Unexpected interrupted-write status")
            committed = bool(np.array_equal(journal.words[slot:slot + SLOT_WORDS], expected))
            journal.reboot()
            boot_ok = int(is_model(journal, post_state if committed else pre_state))
            retry, _ = journal.apply(second)
            retry_ok = int(retry in (OK, DUPLICATE) and is_model(journal, post_state))
            rows.append({**context, "campaign": "power_cut", "mode": "proposed",
                         "fault": fault, "cut": index, "boot_exact": boot_ok,
                         "retry_success": retry_ok, "completed_words": writes})
            if not (boot_ok and retry_ok):
                raise AssertionError(f"Atomicity failed at word {index}, pattern {fault}")

    for mode in ("proposed", "no_idempotency"):
        journal = Journal(packed, words=prefix.words.copy())
        apply_edits(journal, [second, third])
        journal.reboot()
        before = journal.words.copy()
        request = second if mode == "proposed" else Edit(
            4, second.first, second.end, second.from_label, second.to_label
        )
        status, _ = journal.apply(request)
        success = int(status == DUPLICATE and np.array_equal(before, journal.words))
        rows.append({**context, "campaign": "duplicate", "mode": mode, "fault": "lost_ack",
                     "cut": -1, "boot_exact": 1, "retry_success": success, "completed_words": 0})
        if mode == "proposed" and success != 1:
            raise AssertionError("Old retry did not return an idempotent acknowledgement")

    # Change the destination byte after a successful commit. The alternate class remains legal.
    baseline = Journal(packed, words=prefix.words.copy())
    apply_edits(baseline, [second])
    mutated = baseline.words.copy()
    mutated[slot + 2] ^= np.uint32(next(1 << bit for bit in range(8)
                                      if (second.to_label ^ (1 << bit)) < CLASSES
                                      and (second.to_label ^ (1 << bit)) != second.from_label) << 8)
    for mode in ("proposed", "no_crc_validation"):
        journal = Journal(packed, words=mutated.copy(), verify_crc=mode == "proposed")
        boot = int(is_model(journal, pre_state))
        status, _ = journal.apply(second)
        success = int(status in (OK, DUPLICATE) and is_model(journal, post_state))
        rows.append({**context, "campaign": "integrity", "mode": mode, "fault": "payload_bit",
                     "cut": 2, "boot_exact": boot, "retry_success": success, "completed_words": 0})
        if mode == "proposed" and not (boot and success):
            raise AssertionError("A corrupted record was used during replay")

    # Compact twice so the inactive bank contains an older generation. Exercise
    # interrupted erase as well as checkpoint writes while a newer bank survives.
    compact = Journal(packed, words=prefix.words.copy())
    assert compact.compact() == (OK, 1 + CHECKPOINT_WORDS)
    apply_edits(compact, [second])
    assert compact.compact() == (OK, 1 + CHECKPOINT_WORDS)
    for index in range(CHECKPOINT_WORDS + 1):
        for fault in ("before_word", "torn_low_half", "word_complete"):
            journal = Journal(packed, words=compact.words.copy())
            status, writes = journal.compact(cut=index, mask=FAULT_PATTERNS[fault])
            if status != IO:
                raise AssertionError("Expected interrupted checkpoint or erase")
            journal.reboot()
            durable = index == CHECKPOINT_WORDS and fault == "word_complete"
            boot_ok = int(is_model(journal, post_state) and journal.state.generation == 2 + durable)
            if not durable:
                status, _ = journal.compact()
                if status != OK:
                    raise AssertionError("Compaction retry failed")
            journal.reboot()
            retry_ok = int(is_model(journal, post_state) and journal.state.generation == 3
                           and journal.apply(second) == (DUPLICATE, 0))
            if not (boot_ok and retry_ok):
                raise AssertionError(f"Checkpoint recovery failed at operation {index}, {fault}")
            rows.append({**context, "campaign": "compaction", "mode": "proposed",
                         "fault": fault, "cut": index, "boot_exact": boot_ok,
                         "retry_success": retry_ok, "completed_words": writes})

    # Isolate the six-word receipt from the rest of the checkpoint protocol.
    for mode in ("proposed", "no_checkpoint_receipt"):
        journal = Journal(packed, words=prefix.words.copy(), checkpoint_receipt=mode == "proposed")
        apply_edits(journal, [second])
        assert journal.compact()[0] == OK
        journal.reboot()
        boot_ok = int(is_model(journal, post_state))
        success = int(journal.apply(second) == (DUPLICATE, 0))
        assert boot_ok and success == int(mode == "proposed")
        rows.append({**context, "campaign": "checkpoint_retry", "mode": mode,
                     "fault": "lost_ack", "cut": -1, "boot_exact": boot_ok,
                     "retry_success": success, "completed_words": 0})

    # Both layouts use the same erase-before-write and commit-last fault model.
    snapshot = Journal(packed, snapshot_receipt=True)
    assert snapshot.apply(first, snapshot=True) == (OK, 1 + CHECKPOINT_WORDS)
    for index in (0, 1, CHECKPOINT_WORDS):
        for fault in ("before_word", "torn_low_half", "word_complete"):
            journal = Journal(packed, words=snapshot.words.copy(), snapshot_receipt=True)
            status, writes = journal.apply(second, cut=index, mask=FAULT_PATTERNS[fault], snapshot=True)
            assert status == IO
            journal.reboot()
            durable = index == CHECKPOINT_WORDS and fault == "word_complete"
            boot_ok = int(is_model(journal, post_state if durable else pre_state))
            retry = journal.apply(second, snapshot=True)
            expected = (DUPLICATE, 0) if durable else (OK, 1 + CHECKPOINT_WORDS)
            assert boot_ok and retry == expected and is_model(journal, post_state)
            rows.append({**context, "campaign": "snapshot", "mode": "snapshot",
                         "fault": fault, "cut": index, "boot_exact": boot_ok,
                         "retry_success": int(retry[0] in (OK, DUPLICATE)), "completed_words": writes})
    return rows


def stress_campaign(dataset, subject, seed, blocks, labels):
    """Check long legal edit streams against a complete integer reconstruction."""
    rng = np.random.default_rng([seed, subject, 917])
    packed = pack_blocks(blocks, labels, "packed")
    expected = labels.copy()
    journal = Journal(packed, slots=8)
    erases = resets = programmed = 0
    for request in range(1, 65):
        first = int(rng.integers(len(expected)))
        old = int(expected[first])
        limit = first + 1
        while limit < len(expected) and limit - first < 5 and expected[limit] == old:
            limit += 1
        end = int(rng.integers(first + 1, limit + 1))
        new = (old + int(rng.integers(1, CLASSES))) % CLASSES
        edit = Edit(request, first, end, old, new)
        if journal.state.next_slot == journal.slots:
            if request % 2 == 0:
                cut = int(rng.integers(1, CHECKPOINT_WORDS))
                status, done = journal.compact(cut=cut)
                assert status == IO
                erases += 1
                programmed += 4 * (done - 1)
                journal.reboot()
                resets += 1
            assert journal.apply(edit) == (FULL, 0)
            assert journal.compact() == (OK, 1 + CHECKPOINT_WORDS)
            erases += 1
            programmed += 4 * CHECKPOINT_WORDS
        if request % 3 == 0:
            cut = int(rng.integers(6))
            status, done = journal.apply(edit, cut=cut)
            assert status == IO
            programmed += 4 * done
            journal.reboot()
            resets += 1
            assert is_model(journal, (*rebuild(blocks, expected), expected))
            if journal.state.next_slot == journal.slots:
                assert journal.compact() == (OK, 1 + CHECKPOINT_WORDS)
                erases += 1
                programmed += 4 * CHECKPOINT_WORDS
        assert journal.apply(edit) == (OK, SLOT_WORDS)
        programmed += 4 * SLOT_WORDS
        expected[first:end] = new
        if request % 4 == 0:
            journal.reboot()
            resets += 1
        assert journal.apply(edit) == (DUPLICATE, 0)
        assert is_model(journal, (*rebuild(blocks, expected), expected))
    return {"dataset": dataset, "subject": subject, "seed": seed, "verified_edits": 64,
            "exact_retry_acks": 64, "resets": resets, "erase_calls": erases,
            "completed_program_bytes": programmed}


def run(data_root=DEFAULT_DATA, out=RESULTS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_main, all_class, all_conf, all_fault, all_data, all_stress = [], [], [], [], [], []
    for dataset in DATASETS:
        data = prepare(dataset, data_root)
        for subject in np.unique(data["subject"]):
            subject = int(subject)
            source = data["subject"] != subject
            test = (data["subject"] == subject) & data["test"]
            source_x = data["raw"][source].reshape(-1, DIM)
            source_y = np.repeat(data["label"][source], VECTORS)
            metric = fit_metric(source_x)
            sums, counts = statistics(metric.apply(source_x), source_y)
            prior = centroids(sums, counts, np.zeros((CLASSES, DIM), np.int16), 0)
            x = metric.apply(data["raw"][test].reshape(-1, DIM))
            y = np.repeat(data["label"][test], VECTORS)
            if np.any(np.bincount(y, minlength=CLASSES) == 0):
                raise AssertionError(f"Missing test class: {dataset}, {subject}")
            all_data.append({"dataset": dataset, "subject": subject, "source_windows": len(source_y),
                             "calibration_windows": CLASSES * BLOCKS_PER_CLASS * VECTORS,
                             "test_windows": len(y), "test_trials": len(np.unique(data["trial"][test]))})
            for seed in SEEDS:
                selection = calibration_indices(data, subject, np.random.default_rng([seed, subject]))
                blocks = metric.apply(data["raw"][selection])
                labels = data["label"][selection].astype(np.uint8)
                for length in LENGTHS:
                    main, cls, conf, corrupted, edits = evaluate_case(
                        dataset, subject, seed, length, blocks, labels, x, y, prior
                    )
                    all_main.extend(main)
                    all_class.extend(cls)
                    all_conf.extend(conf)
                    if length == 2:
                        all_fault.extend(fault_campaign(dataset, subject, seed, blocks, corrupted, edits))
                        all_stress.append(stress_campaign(dataset, subject, seed, blocks, labels))
        print(f"{dataset}: finished {len(np.unique(data['subject']))} held-out participants × {len(SEEDS)} seeds",
              flush=True)
    for name, records in (("scores", all_main), ("per_class", all_class), ("confusion", all_conf),
                          ("faults", all_fault), ("dataset_inventory", all_data),
                          ("stress", all_stress)):
        pd.DataFrame(records).to_csv(out / f"{name}.csv", index=False, float_format="%.9f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
