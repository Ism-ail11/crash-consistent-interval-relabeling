"""Generate, execute, validate, and export the three reproducibility notebooks."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import os

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from IPython.core.interactiveshell import InteractiveShell
from jupyter_client.jsonutil import json_clean

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

SETUP = """from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "har").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text):
    return nbf.v4.new_code_cell(text.strip())


def dataset_notebook():
    return [
        md("""# 01 · Datasets and fixed-point HAR
Public accelerometer and gyroscope recordings drive the entire experiment. HAPT supplies waist-mounted smartphone IMU recordings. MotionSense supplies pocket-phone Core Motion data. The experiment uses six activity classes per dataset. HAPT transition annotations are excluded from the six-class benchmark.

The target participant contributes only an early trial to calibration. Test windows come from separate later trials. The source feature normalization and prior class prototypes use other participants only. The three seeds change calibration excerpts and simulated annotation mistakes."""),
        py(SETUP + """
from har.data import DATASETS, FILES, LABELS, prepare, segments, windows_of
from har.experiments import SEEDS, calibration_indices
from har.model import DIM, VECTORS, fit_metric
from har.native import c_embeddings, features

print("Seeds:", SEEDS)
display(pd.read_csv(RESULTS / "tables" / "dataset_inventory.csv"))
"""),
        md("""## Recompute the input pipeline
The loader verifies both archives against pinned SHA-256 hashes. The C encoder computes 24 fixed-point features from 100 samples. Windows advance by 25 samples and form four-vector editing blocks."""),
        py("""
data = {name: prepare(name) for name in DATASETS}
inventory = []
for name, values in data.items():
    candidates = np.unique(values["subject"])
    for subject in candidates:
        calibration = calibration_indices(values, int(subject), np.random.default_rng([17, subject]))
        assert not values["test"][calibration].any()
        assert set(values["trial"][calibration]).isdisjoint(
            values["trial"][(values["subject"] == subject) & values["test"]]
        )
    inventory.append({
        "Dataset": name,
        "Subjects": len(candidates),
        "Raw feature windows": values["raw"].shape[0] * VECTORS,
        "Archive SHA-256": FILES[name][2],
    })
display(pd.DataFrame(inventory))
"""),
        py("""
rows = []
for name, values in data.items():
    source = values["subject"] != 1
    metric = fit_metric(values["raw"][source].reshape(-1, DIM))
    segment = next(s for s in segments(name) if s.subject == 1 and s.test and len(s.samples) >= 175)
    windows = windows_of(segment.samples)[:4]
    encoded = features(windows)
    np.testing.assert_array_equal(metric.apply(encoded), c_embeddings(encoded, metric))
    for i, activity in enumerate(LABELS[name]):
        rows.append({"Dataset": name, "Activity": activity,
                     "Test windows": int(np.sum(values["test"] & (values["label"] == i))) * VECTORS})
display(pd.DataFrame(rows).pivot(index="Activity", columns="Dataset", values="Test windows").fillna(0).astype(int))
print("The Python and C integer embeddings agree exactly on both raw-data fixtures.")
"""),
        md("""Dataset archives and their terms are documented in [DATASETS.md](../DATASETS.md). Recomputing every participant–seed run takes a single command from the repository root.

```bash
python -m har.experiments
```"""),
    ]


def journal_notebook():
    return [
        md("""# 02 · Recovery and Cortex-M4 execution
The firmware stores 44-byte packed calibration summaries and appends six-word journal transactions. A two-bank checkpoint preserves the newest complete generation and the most recent committed request, so its retry can still be acknowledged after compaction. This notebook reruns the C tests and checks the saved Renode execution trace."""),
        py(SETUP + """
import subprocess
from har.model import DIM, VECTORS, rebuild
from har.native import DUPLICATE, IO, OK, Edit, Journal, pack_blocks

test_run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=ROOT, text=True, capture_output=True, check=True)
print(test_run.stderr.strip())
"""),
        md("""## Walk through an interrupted edit
The first three flash words are programmed. The fourth word is interrupted. After a fresh C state is reconstructed, the original class sums are recovered. Retrying commits the entire interval. A second retry returns the saved acknowledgement without another write."""),
        py("""
rng = np.random.default_rng(17)
x = rng.integers(-100, 101, size=(12, VECTORS, DIM), dtype=np.int16)
labels = np.repeat(np.arange(6, dtype=np.uint8), 2)
edit = Edit(1, 0, 2, 0, 1)
journal = Journal(pack_blocks(x, labels, "packed"), slots=4)
assert journal.apply(edit, cut=3, mask=0xFFFF)[0] == IO
journal.reboot()
np.testing.assert_array_equal(journal.labels(), labels)
assert journal.apply(edit) == (OK, 6)
journal.reboot()
assert journal.apply(edit) == (DUPLICATE, 0)
assert journal.compact()[0] == OK
journal.reboot()
assert journal.apply(edit) == (DUPLICATE, 0)
fixed = labels.copy()
fixed[:2] = 1
np.testing.assert_array_equal(journal.stats()[0], rebuild(x, fixed)[0])
print("Recovered class counts:", journal.stats()[1].tolist())
"""),
        md("""## Simulator checks and memory
The saved Renode runs compile and execute the same C core on a Cortex-M4 platform with the STM32F4 flash controller model. Three fixtures per dataset enroll from raw sensor windows and exercise reset and recovery. Instruction counts describe emulated work. Linker-section RAM excludes the test mailbox and stack."""),
        py("""
if os.environ.get("HAR_RERUN_RENODE") == "1":
    from har.emulator import run as run_renode
    run_renode(renode=os.environ.get("RENODE", "renode"),
               cross=os.environ.get("CROSS_COMPILE", "arm-none-eabi-"))

renode = pd.read_csv(RESULTS / "renode.csv")
assert (renode.loc[renode.case.str.endswith("_recovery"), "status"] == OK).all()
assert (renode.loc[renode.case == "clean_edit", "program_bytes"] == 24).all()
assert (renode.loc[renode.case == "compacted_latest", "status"] == DUPLICATE).all()
display(pd.read_csv(RESULTS / "tables" / "firmware_cost.csv"))
display(pd.read_csv(RESULTS / "firmware_memory.csv"))
display(Image(filename=str(RESULTS / "figures" / "fig1_recovery.png"), width=910))
"""),
        py("""
display(pd.read_csv(RESULTS / "tables" / "recovery.csv"))
reset_rows = renode.case.str.fullmatch(r"(?:edit|compact|snapshot)_cut_[0-9]+_(?:before|torn_low_half|after)")
display(renode.loc[reset_rows].groupby("layout").size().rename("Cortex-M4 resets").to_frame())
display(pd.read_csv(RESULTS / "tables" / "stress_sequences.csv"))
"""),
        md("""Re-run the Cortex-M4 campaign with Renode and the Arm GNU toolchain on the PATH.

```bash
python -m har.emulator
```"""),
    ]


def results_notebook():
    return [
        md("""# 03 · Results, ablations, and figures
The primary comparison uses two-second annotation errors. Confidence intervals resample held-out participants after averaging the three seeds for each participant. The analysis keeps source-only, uncorrected, and full-vector baselines visible alongside the component ablations."""),
        py(SETUP + """
from har.experiments import SEEDS, run as run_experiments
from har.report import participant_scores, quality_summary

if os.environ.get("HAR_RERUN_DATA") == "1":
    run_experiments()
scores = pd.read_csv(RESULTS / "scores.csv")
assert sorted(scores.seed.unique()) == sorted(SEEDS)
assert set(scores.dataset) == {"HAPT", "MotionSense"}
full = scores[scores["mode"] == "full_vectors"]
assert (full.prediction_agreement == 1).all()
paired = quality_summary(participant_scores(scores))
display(pd.read_csv(RESULTS / "tables" / "quality_primary.csv"))
"""),
        md("""## Primary comparisons and sensitivity
The shaded band shows the participant-bootstrap interval around the proposed method. The two-second setting is the primary comparison. One- and three-second errors are sensitivity analyses."""),
        py("""
for filename in ("fig2_paired_effects.png", "fig3_error_duration.png", "fig5_confusion_change.png"):
    display(Image(filename=str(RESULTS / "figures" / filename), width=920))
"""),
        md("""## What each ablation removes
The full-vector implementations give exact references. The component tests exercise label subtraction, the source prior, history length, edit granularity, packing, retry IDs, CRC validation, and acknowledgement after checkpointing."""),
        py("""
ablation = pd.DataFrame([
    ("Keep wrong labels", "No edit applied"),
    ("Add corrected labels only", "Leave the old class contribution in place"),
    ("Remove source prior", "Use the corrected target-user means"),
    ("8-block edit history", "Reject edits touching earlier blocks"),
    ("2-second edit grid", "Accept only complete two-block pairs"),
    ("Unpacked summaries", "Store 32-bit sums with the same journal and predictions"),
    ("Remove retry IDs", "Treat a repeated request as a new edit"),
    ("Remove CRC validation", "Accept structurally valid corrupt payloads"),
    ("Remove checkpoint receipt", "Keep the high-water ID and discard the latest payload"),
], columns=["Ablation", "Change"])
display(ablation)
display(Image(filename=str(RESULTS / "figures" / "fig4_firmware_cost.png"), width=920))
display(Image(filename=str(RESULTS / "figures" / "fig6_journal_write_cost.png"), width=470))
"""),
        py("""
primary = paired[paired.edit_length_s == 2].pivot(index="dataset", columns="mode", values="macro_f1_pct")
delta = (primary.proposed - primary.source_only).rename("Journal minus source-only (pp)")
display(delta.to_frame().round(2))
display(pd.read_csv(RESULTS / "tables" / "paired_primary.csv").round(2))
print(f"Exact full-vector prediction agreement across {len(full)} participant × seed × duration cases.")
"""),
        md("""Use the repository's SVG or PNG exports for figures and the CSV or LaTeX table for exact values. Recognition quality reflects the simulated label-error workload. Simulator instruction counts and flash operations characterize the implementation under the stated model."""),
    ]


def execute_in_process(notebook):
    """Use IPython's display formatter and cell interpreter without a socket transport."""
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        for number, cell in enumerate((c for c in notebook.cells if c.cell_type == "code"), 1):
            outputs = []

            def publish(data, metadata=None, **_):
                outputs.append(nbf.v4.new_output(
                    "display_data", data=json_clean(data), metadata=json_clean(metadata or {})
                ))

            shell.display_pub.publish = publish
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                run = shell.run_cell(cell.source, store_history=False, silent=True)
            error = run.error_before_exec or run.error_in_exec
            if error is not None:
                raise RuntimeError(f"Notebook cell {number} failed:\n{stdout.getvalue()}") from error
            for channel, stream in (("stdout", stdout), ("stderr", stderr)):
                if stream.getvalue():
                    outputs.insert(0, nbf.v4.new_output("stream", name=channel, text=stream.getvalue()))
            cell.execution_count = number
            cell.outputs = outputs
    finally:
        os.chdir(previous)
        InteractiveShell.clear_instance()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--in-process", action="store_true",
                        help="Execute with IPython in environments that restrict local sockets")
    args = parser.parse_args()
    NOTEBOOKS.mkdir(exist_ok=True)
    for name, cells in (("01_data_and_features", dataset_notebook()),
                        ("02_journal_and_renode", journal_notebook()),
                        ("03_results_and_ablations", results_notebook())):
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        })
        if args.in_process:
            execute_in_process(notebook)
        else:
            NotebookClient(notebook, timeout=300, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.validate(notebook)
        path = NOTEBOOKS / f"{name}.ipynb"
        nbf.write(notebook, path)
        if not args.no_html:
            exporter = HTMLExporter(template_name="lab")
            html, _ = exporter.from_notebook_node(notebook)
            path.with_suffix(".html").write_text(html, encoding="utf-8")
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code)
        assert all(output.output_type != "error" for cell in code for output in cell.outputs)
        print(f"{path.name}: {len(code)} executed code cells, zero errors", flush=True)


if __name__ == "__main__":
    main()
