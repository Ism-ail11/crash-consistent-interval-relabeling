"""Run compiled Cortex-M4 firmware and verify actual reset/replay in Renode."""

import argparse
from dataclasses import asdict
from pathlib import Path
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .data import DATASETS, DEFAULT_DATA, prepare, segments, windows_of
from .experiments import RESULTS, SEEDS, calibration_indices, make_edits
from .model import CLASSES, DIM, PRIOR_COUNT, VECTORS, centroids, fit_metric, predict, rebuild, statistics
from .native import (
    CHECKPOINT_WORDS, SNAPSHOT_WORDS, CONFLICT, DUPLICATE, OK, ROOT, STALE,
    Edit, features, pack_blocks,
)

MAILBOX = (
    "magic", "request", "response", "command", "status", "layout", "epoch", "block_count",
    "edit_floor", "flags", "edit_id", "first", "end", "from", "to", "cut", "mask", "operations",
    "program_bytes", "erase_count", "state_hash", "last_id", "generation", "bank", "next_slot",
    "prediction", "prior_count", "boot_count", "stage", "scratch_bytes",
)
PROVISION, OPEN, EDIT, COMPACT, INFER, ENROLL, REBUILD, SNAPSHOT_EDIT, CACHE_BUILD = range(1, 10)
TARGET_LAYOUTS = {"summary": 0, "vectors": 1, "packed": 2, "snapshot": 2}
MASKS = {"before": 0, "torn_low_half": 0x0000FFFF, "after": 0xFFFFFFFF}
PRINTED = ("magic", "response", "status", "operations", "program_bytes", "erase_count",
           "state_hash", "last_id", "generation", "bank", "next_slot", "prediction", "boot_count",
           "scratch_bytes")
UINT32 = 0xFFFFFFFF


def model_hash(blocks, labels):
    sums, counts = rebuild(blocks, labels)
    value = 2166136261
    for c in range(CLASSES):
        for word in (counts[c], *sums[c]):
            value = ((value ^ (int(word) & UINT32)) * 16777619) & UINT32
    for label in labels:
        value = ((value ^ int(label)) * 16777619) & UINT32
    return value


def symbols(binary, cross):
    lines = subprocess.check_output([f"{cross}nm", "-S", str(binary)], text=True).splitlines()
    addresses, sizes = {}, {}
    for line in lines:
        items = line.split()
        if len(items) == 4:
            address, size, _, name = items
            addresses[name] = int(address, 16)
            sizes[name] = int(size, 16)
    return addresses, sizes


class Script:
    def __init__(self, binary, addresses, directory):
        self.addresses = addresses
        self.directory = directory
        self.text = [
            'mach create "har"',
            'machine LoadPlatformDescription @firmware/stm32f4-minimal.repl',
            f'sysbus LoadELF @{binary.relative_to(ROOT).as_posix()}',
            'macro reset',
            '"""',
            'cpu VectorTableOffset 0x08000000',
            '"""',
            'runMacro $reset',
            'logLevel 3 flash_controller',
        ]
        for name in ("operation_begin", "operation_end"):
            self.text.append(
                f'cpu AddHook 0x{addresses[name]:x} "self.InfoLog(\'{name} {{0}}\', self.ExecutedInstructions)"'
            )
        self.text.append('emulation RunFor "0.004"')
        self.seq = 0
        self.expected = []

    def load(self, symbol, array):
        source = self.directory / f"{symbol}-{len(self.text)}.bin"
        source.write_bytes(np.ascontiguousarray(array).tobytes())
        self.text.append(f'sysbus LoadBinary @{source.relative_to(ROOT).as_posix()} 0x{self.addresses[symbol]:x}')

    def write(self, name, value):
        addr = self.addresses["mailbox"] + 4 * MAILBOX.index(name)
        self.text.append(f"sysbus WriteDoubleWord 0x{addr:x} 0x{int(value) & UINT32:x}")

    def send(self, code, tag, *, checks=None, read_block=False, **fields):
        self.seq += 1
        fields = {"command": code, "cut": -1, "mask": 0, **fields}
        for name, value in fields.items():
            self.write(name, value)
        self.write("request", self.seq)
        self.text.append('emulation RunFor "0.004"')
        for name in PRINTED:
            addr = self.addresses["mailbox"] + 4 * MAILBOX.index(name)
            self.text.append(f"sysbus ReadDoubleWord 0x{addr:x}")
        if read_block:
            for word in range(25):
                self.text.append(f"sysbus ReadDoubleWord 0x{self.addresses['output_block'] + 4 * word:x}")
        self.expected.append((self.seq, tag, checks or {}, read_block, fields["cut"] >= 0))

    def edit(self, edit, tag, **fields):
        args = asdict(edit)
        args["edit_id"] = args.pop("id")
        args["from"] = args.pop("from_label")
        args["to"] = args.pop("to_label")
        code = SNAPSHOT_EDIT if fields.pop("snapshot", False) else EDIT
        self.send(code, tag, **fields, **args)

    def file(self):
        source = self.directory / "run.resc"
        source.write_text("\n".join([*self.text, "quit", ""]), encoding="utf-8")
        return source


def dataset_fixture(dataset, data, data_root, subject, seed):
    source = data["subject"] != subject
    raw = data["raw"][source].reshape(-1, DIM)
    metric = fit_metric(raw)
    sums, counts = statistics(metric.apply(raw), np.repeat(data["label"][source], VECTORS))
    prior = centroids(sums, counts, np.zeros((CLASSES, DIM), np.int16), 0)
    selection = calibration_indices(data, subject, np.random.default_rng([seed, subject]))
    blocks = metric.apply(data["raw"][selection])
    labels = data["label"][selection].astype(np.uint8)
    corrupted, edits = make_edits(labels, dataset, np.random.default_rng([seed, subject, 2]), 2)
    raw_segments = {(s.trial, s.group): windows_of(s.samples)
                    for s in segments(dataset, data_root, subject=subject)}
    calibration_samples = []
    for row in selection:
        key = int(data["trial"][row]), int(data["group"][row])
        start = int(data["block_index"][row]) * VECTORS
        calibration_samples.append(raw_segments[key][start:start + VECTORS])
    test_rows = np.flatnonzero((data["subject"] == subject) & data["test"])
    first = test_rows[0]
    samples = raw_segments[int(data["trial"][first]), int(data["group"][first])][:VECTORS]
    return metric, prior, blocks, labels, corrupted, edits, samples, calibration_samples


def run_script(script, renode):
    source = script.file()
    proc = subprocess.run([renode, "--plain", "--disable-gui", "--console", str(source.relative_to(ROOT))],
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=360, cwd=ROOT, check=False)
    log = proc.stdout.replace("\r", "")
    (script.directory / "monitor.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0 or "There was an error executing command" in log or "[ERROR]" in log:
        raise RuntimeError(f"Renode failed. Tail of the monitor log:\n{log[-6000:]}")
    # Console and logger share stdout. A log message can split a printed word.
    monitor = re.sub(r"\[\d\d:\d\d:\d\d\.\d+\] \[[A-Z]+\] [^\n]*\n", "", log)
    values = [int(m, 16) for m in re.findall(r"(?m)^\s*(0x[0-9a-fA-F]{8})\s*$", monitor)]
    events = [(key, int(num)) for key, num in re.findall(r"(operation_begin|operation_end) ([0-9]+)", log)]
    results = []
    offset = 0
    event_index = 0
    for seq, tag, checks, read_block, reset in script.expected:
        result = dict(zip(PRINTED, values[offset:offset + len(PRINTED)], strict=True))
        offset += len(PRINTED)
        if result["response"] != seq or result["magic"] != 0x48415234:
            raise AssertionError(f"{tag}: target did not acknowledge request {seq}: {result}")
        if result["status"] > 2**31 - 1:
            result["status"] -= 2**32
        if read_block:
            result["block"] = np.asarray(values[offset:offset + 25], dtype="<u4").tobytes()
            offset += 25
        if event_index >= len(events) or events[event_index][0] != "operation_begin":
            raise AssertionError(f"Missing instruction counter at start of {tag}")
        before = events[event_index][1]
        event_index += 1
        if not reset:
            if event_index >= len(events) or events[event_index][0] != "operation_end":
                raise AssertionError(f"Missing instruction counter at end of {tag}")
            result["instructions"] = events[event_index][1] - before
            if result["instructions"] < 0:
                raise AssertionError(f"Invalid instruction count for {tag}")
            event_index += 1
        else:
            result["instructions"] = None
        for field, expected in checks.items():
            if result[field] != expected:
                raise AssertionError(f"{tag}: {field} was {result[field]!r}, expected {expected!r}")
        result.update({"case": tag, "sequence": seq})
        result.pop("block", None)
        results.append(result)
    if offset != len(values) or event_index != len(events):
        raise AssertionError(f"Unconsumed trace data: {len(values) - offset} words, {len(events) - event_index} hooks")
    return results


def provision_fixture(script, blocks, layout, state_hash):
    script.send(PROVISION, "provision", layout=TARGET_LAYOUTS[layout], epoch=17,
                block_count=len(blocks), flags=4 if layout == "snapshot" else 0, checks={"status": OK})
    script.send(OPEN, "open_initial", checks={"status": OK, "state_hash": state_hash})


def stage_calibration(script, fixture, layout, enroll=False):
    metric, prior, blocks, clean, initial, edits, samples, calibration_samples = fixture
    script.load("input_metric", np.concatenate((metric.center, metric.scale)).astype("<i4"))
    script.load("input_prior", prior.astype("<i2"))
    packed = pack_blocks(blocks, initial, "packed" if layout == "snapshot" else layout)
    if enroll:
        for i, windows in enumerate(calibration_samples):
            script.load("input_samples", windows.astype("<i2"))
            summary = pack_blocks(blocks[i:i + 1], initial[i:i + 1])[0].tobytes()
            script.send(ENROLL, "enroll_calibration", stage=i, layout=TARGET_LAYOUTS[layout],
                        **{"from": initial[i]}, checks={"status": OK, "block": summary}, read_block=True)
    else:
        script.load("input_base", packed)
    script.load("input_samples", samples.astype("<i2"))
    embeddings = metric.apply(features(samples))
    reference = pack_blocks(embeddings[None, ...], np.asarray([1], np.uint8))[0].tobytes()
    script.send(ENROLL, "enroll_test", stage=UINT32, **{"from": 1},
                checks={"status": OK, "block": reference}, read_block=True)
    return embeddings


def snapshot_script(script, fixture):
    _, prior, blocks, clean, initial, edits, _, _ = fixture
    embeddings = stage_calibration(script, fixture, "snapshot")
    initial_hash = model_hash(blocks, initial)
    corrected_hash = model_hash(blocks, clean)
    provision_fixture(script, blocks, "snapshot", initial_hash)
    for edit in edits:
        script.edit(edit, "snapshot_edit", snapshot=True,
                    checks={"status": OK, "program_bytes": CHECKPOINT_WORDS * 4, "erase_count": 1})
    script.send(OPEN, "open_corrected", checks={"status": OK, "state_hash": corrected_hash, "generation": 3})
    sums, counts = rebuild(blocks, clean)
    predicted = int(predict(embeddings[:1], centroids(sums, counts, prior, PRIOR_COUNT))[0])
    script.send(INFER, "infer", prior_count=PRIOR_COUNT, checks={"status": OK, "prediction": predicted})
    script.edit(edits[-1], "snapshot_lost_ack", snapshot=True,
                checks={"status": DUPLICATE, "program_bytes": 0})
    first = edits[0]
    updated = initial.copy()
    updated[first.first:first.end] = first.to_label
    updated_hash = model_hash(blocks, updated)
    for cut in (0, CHECKPOINT_WORDS, CHECKPOINT_WORDS + 1):
        for name in ("before", "after") if cut != CHECKPOINT_WORDS + 1 else ("after",):
            provision_fixture(script, blocks, "snapshot", initial_hash)
            case = f"snapshot_cut_{cut}_{name}"
            boot_before = 1 + sum(old[4] for old in script.expected)
            script.edit(first, case, snapshot=True, cut=cut, mask=MASKS[name],
                        checks={"boot_count": boot_before + 1})
            durable = cut >= CHECKPOINT_WORDS and name == "after"
            script.send(OPEN, f"{case}_recovery",
                        checks={"status": OK, "state_hash": updated_hash if durable else initial_hash})
            script.edit(first, f"{case}_retry", snapshot=True,
                        checks={"status": DUPLICATE if durable else OK, "state_hash": updated_hash})


def fixture_script(script, fixture, layout):
    if layout == "snapshot":
        snapshot_script(script, fixture)
        return
    _, prior, blocks, clean, initial, edits, _, _ = fixture
    initial_hash = model_hash(blocks, initial)
    corrected_hash = model_hash(blocks, clean)
    embeddings = stage_calibration(script, fixture, layout, enroll=layout == "packed")
    def provision():
        provision_fixture(script, blocks, layout, initial_hash)

    provision()
    for edit in edits:
        script.edit(edit, "clean_edit", checks={"status": OK, "program_bytes": 24})
        if layout == "vectors":
            script.send(REBUILD, "full_rebuild", checks={"status": OK})
    script.send(OPEN, "open_corrected", checks={"status": OK, "state_hash": corrected_hash})
    sums, counts = rebuild(blocks, clean)
    predicted = int(predict(embeddings[:1], centroids(sums, counts, prior, PRIOR_COUNT))[0])
    script.send(INFER, "infer", prior_count=PRIOR_COUNT,
                checks={"status": OK, "prediction": predicted, "state_hash": corrected_hash})
    script.edit(edits[0], "late_duplicate", checks={"status": DUPLICATE, "program_bytes": 0})
    script.send(COMPACT, "compact", checks={"status": OK, "state_hash": corrected_hash,
                                           "generation": 1, "erase_count": 1})
    script.send(OPEN, "open_compacted", checks={"status": OK, "state_hash": corrected_hash, "generation": 1})
    script.edit(edits[0], "compacted_id", checks={"status": STALE, "program_bytes": 0})
    script.edit(edits[-1], "compacted_latest", checks={"status": DUPLICATE, "program_bytes": 0})
    latest = edits[-1]
    conflicting = Edit(latest.id, latest.first, latest.end, latest.from_label,
                       next(c for c in range(CLASSES) if c not in (latest.from_label, latest.to_label)))
    script.edit(conflicting, "compacted_conflict", checks={"status": CONFLICT, "program_bytes": 0})
    reverse = Edit(4, edits[0].first, edits[0].end, edits[0].to_label, edits[0].from_label)
    reverse_labels = clean.copy()
    reverse_labels[reverse.first:reverse.end] = reverse.to_label
    reversed_hash = model_hash(blocks, reverse_labels)
    script.edit(reverse, "edit_after_compaction", checks={"status": OK, "state_hash": reversed_hash})
    script.send(COMPACT, "compact_again", checks={"status": OK, "state_hash": reversed_hash, "generation": 2})
    script.send(OPEN, "open_compacted_again", checks={"status": OK, "state_hash": reversed_hash, "generation": 2})

    if layout == "vectors":
        provision()
        cache_size = len(blocks) * 100
        script.send(CACHE_BUILD, "cached_open_initial",
                    checks={"status": OK, "state_hash": initial_hash, "scratch_bytes": cache_size})
        for edit in edits:
            script.edit(edit, "cached_edit",
                        checks={"status": OK, "program_bytes": 24, "scratch_bytes": cache_size})
        script.send(CACHE_BUILD, "cached_open_corrected",
                    checks={"status": OK, "state_hash": corrected_hash, "scratch_bytes": cache_size})

    if layout != "packed":
        return
    first = edits[0]
    changed = initial.copy()
    changed[first.first:first.end] = first.to_label
    changed_hash = model_hash(blocks, changed)
    for cut in range(7):
        for name, mask in (MASKS.items() if cut < 6 else (("after", MASKS["after"]),)):
            provision()
            case = f"edit_cut_{cut}_{name}"
            boot_before = 1 + sum(old[4] for old in script.expected)
            script.edit(first, case, cut=cut, mask=mask, checks={"boot_count": boot_before + 1})
            durable = cut >= 5 and name == "after"
            script.send(OPEN, f"{case}_recovery",
                        checks={"status": OK, "state_hash": changed_hash if durable else initial_hash})
            script.edit(first, f"{case}_retry", checks={"status": DUPLICATE if durable else OK,
                                                       "state_hash": changed_hash})
    for cut in (0, 1, 4, 1 + CHECKPOINT_WORDS // 2, CHECKPOINT_WORDS - 1, CHECKPOINT_WORDS,
                CHECKPOINT_WORDS + 1):
        for name in ("before", "after") if cut <= CHECKPOINT_WORDS else ("after",):
            provision()
            script.edit(first, "prepare_compaction", checks={"status": OK})
            script.send(COMPACT, "prepare_checkpoint_a", checks={"status": OK, "generation": 1})
            script.send(COMPACT, "prepare_checkpoint_b", checks={"status": OK, "generation": 2})
            case = f"compact_cut_{cut}_{name}"
            boot_before = 1 + sum(old[4] for old in script.expected)
            script.send(COMPACT, case, cut=cut, mask=MASKS[name], checks={"boot_count": boot_before + 1})
            durable = cut >= CHECKPOINT_WORDS and name == "after"
            script.send(OPEN, f"{case}_recovery",
                        checks={"status": OK, "state_hash": changed_hash, "generation": 2 + int(durable)})
            if not durable:
                script.send(COMPACT, f"{case}_retry",
                            checks={"status": OK, "state_hash": changed_hash, "generation": 3})
            script.edit(first, f"{case}_receipt", checks={"status": DUPLICATE, "program_bytes": 0})


def run(renode="renode", cross="arm-none-eabi-", data_root=DEFAULT_DATA, out=RESULTS):
    binary = ROOT / "build" / "har-stm32f4.elf"
    subprocess.run(["make", "firmware", f"CROSS_COMPILE={cross}"], cwd=ROOT, check=True)
    addr, sizes = symbols(binary, cross)
    version = subprocess.check_output([renode, "--version"], text=True).splitlines()[0]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    log_dir = ROOT / "build" / "renode"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in DATASETS:
        data = prepare(dataset, data_root)
        participants = np.unique(data["subject"])
        chosen = participants[np.linspace(0, len(participants) - 1, len(SEEDS), dtype=int)]
        for seed, subject in zip(SEEDS, chosen, strict=True):
            subject = int(subject)
            fixture = dataset_fixture(dataset, data, data_root, subject, seed)
            for layout in ("packed", "summary", "vectors", "snapshot"):
                directory = log_dir / f"{dataset.lower()}-{seed}-{layout}"
                directory.mkdir(exist_ok=True)
                script = Script(binary, addr, directory)
                fixture_script(script, fixture, layout)
                jobs.append((dataset, layout, seed, subject, script))
    with ThreadPoolExecutor(max_workers=3) as pool:
        work = [pool.submit(run_script, script, renode) for *_, script in jobs]
        for (dataset, layout, seed, subject, _), pending in zip(jobs, work, strict=True):
            result = pending.result()
            rows.extend({"dataset": dataset, "layout": layout, "seed": seed,
                         "subject": subject, **row} for row in result)
            print(f"{dataset} seed {seed} {layout}: verified {len(result)} Renode commands", flush=True)
    pd.DataFrame(rows).to_csv(out / "renode.csv", index=False)
    size_table = subprocess.check_output([f"{cross}size", "-A", str(binary)], text=True)
    section_bytes = {name: int(n) for name, n in re.findall(r"(?m)^(\.\w+)\s+(\d+)", size_table)}
    memory = {
        "renode": version,
        "compiler": subprocess.check_output([f"{cross}gcc", "--version"], text=True).splitlines()[0],
        "executable_flash": sum(section_bytes.get(s, 0) for s in (".isr_vector", ".text", ".data")),
        "static_firmware_ram": sum(section_bytes.get(s, 0) for s in (".bss", ".data")),
        "simulator_mailbox_ram": section_bytes[".mailbox"],
        "model_state_ram": sizes["state"],
        "logical_journal_bank_bytes": 4 * (CHECKPOINT_WORDS + 16 * 6),
        "reserved_stm32f4_flash_bytes": 3 * 128 * 1024,
    }
    pd.DataFrame([memory]).to_csv(out / "firmware_memory.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renode", default="renode")
    parser.add_argument("--cross", default="arm-none-eabi-")
    parser.add_argument("--data", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--out", default=RESULTS, type=Path)
    args = parser.parse_args()
    run(args.renode, args.cross, args.data, args.out)


if __name__ == "__main__":
    main()
