"""Validate saved experiment artifacts and optionally emit a reproducibility summary."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from har.data import DATASETS  # noqa: E402
from har.experiments import LENGTHS, MODES, SEEDS  # noqa: E402


def verify():
    results = ROOT / "results"
    scores = pd.read_csv(results / "scores.csv")
    faults = pd.read_csv(results / "faults.csv")
    renode = pd.read_csv(results / "renode.csv")
    memory = pd.read_csv(results / "firmware_memory.csv").iloc[0]
    stress = pd.read_csv(results / "stress.csv")
    assert set(scores["dataset"]) == set(DATASETS)
    assert set(scores["seed"]) == set(SEEDS)
    assert set(scores["edit_length_s"]) == set(LENGTHS)
    assert set(scores["mode"]) == set(MODES)
    assert scores.groupby("dataset")["subject"].nunique().to_dict() == {"HAPT": 30, "MotionSense": 24}
    key = ["dataset", "subject", "seed", "edit_length_s"]
    assert scores.groupby(key).size().eq(len(MODES)).all()
    assert scores["macro_f1"].between(0, 1).all()
    full = scores[scores["mode"] == "full_vectors"].set_index(key)
    proposed = scores[scores["mode"] == "proposed"].set_index(key)
    assert len(full) == len(proposed) == 54 * 3 * 3
    for name in ("full_vectors", "unpacked_summary"):
        assert scores.loc[scores["mode"] == name, "prediction_agreement"].eq(1).all()
    np.testing.assert_array_equal(full["macro_f1"], proposed.loc[full.index, "macro_f1"])
    np.testing.assert_array_equal(full["accuracy"], proposed.loc[full.index, "accuracy"])

    expected_faults = {"checkpoint_retry": 162, "compaction": 22356, "duplicate": 162,
                       "integrity": 162, "power_cut": 4050}
    checked = faults[faults["mode"] == "proposed"]
    assert checked.groupby("campaign").size().to_dict() == expected_faults
    assert checked[["boot_exact", "retry_success"]].eq(1).all().all()
    ablations = faults["mode"].str.startswith("no_")
    assert faults.loc[ablations, "retry_success"].eq(0).all()
    snapshots = faults.loc[faults["mode"] == "snapshot"]
    assert len(snapshots) == 1458 and snapshots["boot_exact"].eq(1).all()
    assert snapshots["retry_success"].sum() == 1458
    assert len(stress) == 54 * len(SEEDS)
    assert stress[["verified_edits", "exact_retry_acks"]].eq(64).all().all()

    cuts = renode["case"].str.fullmatch(r"(?:edit|compact|snapshot)_cut_[0-9]+_(?:before|torn_low_half|after)")
    proposed_cuts = int((cuts & renode["layout"].eq("packed")).sum())
    assert proposed_cuts == 192 and int(cuts.sum()) == 222
    assert renode.loc[cuts, "instructions"].isna().all()
    assert renode.loc[~cuts, "instructions"].gt(0).all()
    assert renode.loc[renode["case"].str.endswith("_recovery"), "status"].eq(0).all()
    assert renode.loc[renode["case"] == "clean_edit", "program_bytes"].eq(24).all()
    assert renode.loc[renode["case"] == "cached_edit", "scratch_bytes"].eq(2400).all()
    assert renode.loc[renode["case"] == "snapshot_edit", "program_bytes"].eq(180).all()
    assert renode.loc[renode["case"] == "snapshot_edit", "erase_count"].eq(1).all()
    assert renode.loc[renode["case"] == "compacted_latest", "status"].eq(1).all()
    assert set(renode["seed"]) == set(SEEDS)
    for _, frame in renode.groupby("dataset"):
        assert frame["subject"].nunique() == len(SEEDS)
        assert frame.groupby("layout")["sequence"].size().to_dict() == {
            "packed": 251 * len(SEEDS), "summary": 17 * len(SEEDS),
            "vectors": 27 * len(SEEDS), "snapshot": 34 * len(SEEDS),
        }
    assert renode["case"].eq("enroll_calibration").sum() == 24 * len(DATASETS) * len(SEEDS)
    assert memory["model_state_ram"] > 0

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    cells = 0
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
        assert all(output.output_type != "error" for cell in code for output in cell.outputs)
        assert any(output.output_type == "display_data" for cell in code for output in cell.outputs)
        assert path.with_suffix(".html").stat().st_size > 1000
        cells += len(code)

    for extension in ("png", "svg"):
        assert len(list((results / "figures").glob(f"*.{extension}"))) == 6
    assert len(list((results / "tables").glob("*.tex"))) == 3

    primary = proposed.reset_index()
    primary = primary[primary["edit_length_s"] == 2].groupby("dataset")["macro_f1"].mean()
    tracked = sorted(path for path in results.glob("*.csv"))
    return {
        "datasets": list(DATASETS), "seeds": list(SEEDS), "ablation_count": 9,
        "participant_seed_duration_cases": len(full), "full_vector_prediction_agreement": 1.0,
        "primary_macro_f1_pct": (100 * primary).round(4).to_dict(),
        "native_c_exact_recovery_cases": expected_faults,
        "renode_commands_verified": len(renode), "renode_resets_verified": int(cuts.sum()),
        "renode_packed_journal_resets": proposed_cuts,
        "stress_edits_verified": int(stress["verified_edits"].sum()),
        "stress_resets_verified": int(stress["resets"].sum()),
        "renode_version": memory["renode"], "executed_notebooks": len(notebooks),
        "executed_notebook_cells": cells, "notebook_errors": 0,
        "results_csv_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare with the saved validation summary")
    args = parser.parse_args()
    checked = verify()
    output = ROOT / "results" / "verification.json"
    if args.check:
        assert checked == json.loads(output.read_text(encoding="utf-8")), "Saved validation differs"
    else:
        output.write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checked, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
