"""Generate participant-level tables and reproducible paper figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .data import DATASETS, LABELS
from .experiments import RESULTS
from .native import LAYOUTS

INK = "#192b41"
SUBTLE = "#425970"
TEAL = "#087d75"
BLUE = "#325a9f"
ORANGE = "#b5512c"
SAND = "#f7e0d3"
GRID = "#dce5ec"
PAPER = "#ffffff"
NAMES = {
    "source_only": "Source only", "no_correction": "Keep wrong labels",
    "add_only": "Add corrected labels only", "no_prior": "Remove source prior",
    "short_history": "8-block edit history", "coarse_blocks": "2-second edit grid",
    "full_vectors": "Full-vector reference", "unpacked_summary": "Unpacked summaries",
    "proposed": "Packed journal",
    "no_idempotency": "Remove retry IDs", "no_crc_validation": "Remove CRC validation",
    "no_checkpoint_receipt": "Remove checkpoint receipt", "snapshot": "Two-bank snapshot",
}
CAMPAIGNS = {
    "power_cut": "Edit interrupted",
    "compaction": "Compaction interrupted",
    "duplicate": "Lost acknowledgement",
    "integrity": "Corrupted payload",
    "checkpoint_retry": "Lost acknowledgement across checkpoint", "snapshot": "Snapshot interrupted",
}
SCORE_ORDER = tuple(NAMES)[:9]
SYSTEM_NAMES = {
    "packed": "Packed journal",
    "summary": "Unpacked journal",
    "vectors": "Incremental vectors",
    "vectors_cached": "RAM-cached vectors",
    "vectors_rebuild": "Full-vector rebuild",
    "snapshot": "Two-bank snapshot",
}


def bootstrap(values, seed=314159, draws=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    lo, hi = np.quantile(sampled, [0.025, 0.975])
    return float(lo), float(hi)


def participant_scores(scores):
    return scores.groupby(["dataset", "subject", "edit_length_s", "mode"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"),
        prediction_agreement=("prediction_agreement", "mean")
    )


def quality_summary(people):
    rows = []
    for (dataset, length), group in people.groupby(["dataset", "edit_length_s"], sort=False):
        ref = group[group["mode"] == "no_correction"].set_index("subject")["macro_f1"]
        for mode in SCORE_ORDER:
            frame = group[group["mode"] == mode].set_index("subject")
            score = 100 * frame["macro_f1"]
            delta = 100 * (frame["macro_f1"] - ref)
            lower, upper = bootstrap(score)
            dlo, dhi = bootstrap(delta)
            rows.append({
                "dataset": dataset, "edit_length_s": int(length), "mode": mode,
                "participants": len(frame), "macro_f1_pct": score.mean(), "ci_low": lower,
                "ci_high": upper, "accuracy_pct": 100 * frame["accuracy"].mean(),
                "delta_vs_uncorrected_pp": delta.mean(), "delta_ci_low": dlo, "delta_ci_high": dhi,
                "agreement_with_proposed_pct": 100 * frame["prediction_agreement"].mean(),
            })
    return pd.DataFrame(rows)


def fault_summary(faults):
    return faults.groupby(["dataset", "campaign", "mode"], as_index=False).agg(
        cases=("boot_exact", "size"), exact_boots=("boot_exact", "sum"), successful_retries=("retry_success", "sum")
    )


def firmware_summary(renode, data):
    rows = []
    for (dataset, layout), group in renode.groupby(["dataset", "layout"], sort=False):
        is_snapshot = layout == "snapshot"
        clean = group[group["case"] == ("snapshot_edit" if is_snapshot else "clean_edit")]
        def typical(case, field="instructions"):
            values = group.loc[group["case"] == case, field]
            return int(values.median()) if len(values) else None

        dtype = LAYOUTS["packed" if is_snapshot else layout]
        people = data.loc[data["dataset"] == dataset]
        blocks = int(people["calibration_windows"].iloc[0]) // 4
        row = {
            "dataset": dataset, "layout": layout,
            "calibration_payload_bytes": dtype.itemsize * blocks,
            "base_image_program_bytes": typical("provision", "program_bytes"),
            "edit_program_bytes": int(clean["program_bytes"].median()),
            "edit_erases": int(clean["erase_count"].median()),
            "cache_ram_bytes": int(clean["scratch_bytes"].median()),
            "correction_instructions_median": int(clean["instructions"].median()),
            "initial_open_instructions": typical("open_initial"),
            "compacted_open_instructions": typical("open_compacted"),
            "edit_replay_instructions": typical("open_corrected"),
            "checkpoint_program_bytes": typical("compact", "program_bytes"),
            "checkpoint_instructions": typical("compact"),
            "inference_instructions": typical("infer"),
        }
        rows.append(row)
        if layout == "vectors":
            timings = clean["instructions"].to_numpy()
            rebuild = group.loc[group["case"] == "full_rebuild", "instructions"].to_numpy()
            rows.append({**row, "layout": "vectors_rebuild",
                         "correction_instructions_median": int(np.median(timings + rebuild))})
            rows.append({**row, "layout": "vectors_cached",
                         "correction_instructions_median": typical("cached_edit"),
                         "cache_ram_bytes": typical("cached_edit", "scratch_bytes"),
                         "initial_open_instructions": typical("cached_open_initial"),
                         "edit_replay_instructions": typical("cached_open_corrected"),
                         "compacted_open_instructions": None, "checkpoint_program_bytes": None,
                         "checkpoint_instructions": None, "inference_instructions": None})
    return pd.DataFrame(rows)


def fmt_ci(value, low, high, signed=False):
    number = f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{number} [{low:.2f}, {high:.2f}]"


def latex_table(path, headers, rows, columns):
    def escape(value):
        return str(value).replace("%", r"\%").replace("_", r"\_")

    row_end = r" \\"
    lines = [
        r"\begin{tabular}{" + columns + "}", r"\toprule",
        " & ".join(headers) + row_end, r"\midrule",
        *(" & ".join(escape(value) for value in row) + row_end for row in rows),
        r"\bottomrule", r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_tables(out, quality, faults, systems, data, scores, stress):
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    primary = quality[quality["edit_length_s"] == 2].copy()
    primary["Method"] = primary["mode"].map(NAMES)
    primary["Macro-F1 % [95% CI]"] = primary.apply(
        lambda r: fmt_ci(r.macro_f1_pct, r.ci_low, r.ci_high), axis=1
    )
    primary["Δ vs wrong labels (pp)"] = primary.apply(
        lambda r: fmt_ci(r.delta_vs_uncorrected_pp, r.delta_ci_low, r.delta_ci_high, True), axis=1
    )
    quality_view = primary.rename(columns={"dataset": "Dataset"})[
        ["Dataset", "Method", "Macro-F1 % [95% CI]", "Δ vs wrong labels (pp)"]
    ]
    fault_view = faults.copy()
    fault_view["Campaign"] = fault_view["campaign"].map(CAMPAIGNS)
    fault_view["Method"] = fault_view["mode"].map(NAMES)
    fault_view["Exact boot"] = fault_view.apply(lambda r: f"{r.exact_boots}/{r.cases}", axis=1)
    fault_view["Correct outcome after retry"] = fault_view.apply(
        lambda r: f"{r.successful_retries}/{r.cases}", axis=1
    )
    fault_view = fault_view.rename(columns={"dataset": "Dataset"})[
        ["Dataset", "Campaign", "Method", "Exact boot", "Correct outcome after retry"]
    ]
    systems_view = systems.assign(layout=systems["layout"].map(SYSTEM_NAMES)).rename(columns={
        "dataset": "Dataset", "layout": "Implementation",
        "calibration_payload_bytes": "Calibration (B)", "edit_program_bytes": "Edit (B)",
        "edit_erases": "Edit erases",
        "cache_ram_bytes": "Extra cache RAM (B)",
        "correction_instructions_median": "Edit path (insn)",
        "edit_replay_instructions": "Boot after 3 edits (insn)",
    })[["Dataset", "Implementation", "Calibration (B)", "Extra cache RAM (B)", "Edit (B)", "Edit erases",
        "Edit path (insn)", "Boot after 3 edits (insn)"]]
    inventory = data.groupby("dataset", as_index=False).agg(
        Participants=("subject", "nunique"), **{"Test windows": ("test_windows", "sum"),
                                                "Calibration windows/person": ("calibration_windows", "first")}
    ).rename(columns={"dataset": "Dataset"})
    seed_table = scores[scores["edit_length_s"] == 2].groupby(
        ["dataset", "seed", "mode"], as_index=False
    ).agg(macro_f1_pct=("macro_f1", lambda x: 100 * x.mean()))
    stress_view = stress.groupby("dataset", as_index=False).agg(
        streams=("seed", "size"), edits=("verified_edits", "sum"), retry_acks=("exact_retry_acks", "sum"),
        resets=("resets", "sum"), erase_calls=("erase_calls", "sum"),
        programmed_bytes=("completed_program_bytes", "sum"),
    ).rename(columns={"dataset": "Dataset"})
    primary_people = participant_scores(scores).query("edit_length_s == 2")
    comparisons = []
    for dataset, frame in primary_people.groupby("dataset"):
        pivot = frame.pivot(index="subject", columns="mode", values="macro_f1")
        for mode in ("no_correction", "source_only", "add_only"):
            values = 100 * (pivot["proposed"] - pivot[mode])
            low, high = bootstrap(values)
            comparisons.append({"dataset": dataset, "reference": mode, "participants": len(values),
                                "delta_pp": values.mean(), "ci_low": low, "ci_high": high})

    for name, frame in (("quality_primary", quality_view), ("recovery", fault_view),
                        ("firmware_cost", systems_view), ("dataset_inventory", inventory),
                        ("stress_sequences", stress_view)):
        frame.to_csv(tables / f"{name}.csv", index=False)
        (tables / f"{name}.md").write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    for name, frame in (("quality_all_effects", quality), ("recovery_counts", faults),
                        ("firmware_numeric", systems), ("seed_breakdown", seed_table),
                        ("paired_primary", pd.DataFrame(comparisons))):
        frame.to_csv(tables / f"{name}.csv", index=False, float_format="%.6f")

    wide = primary.pivot(index="mode", columns="dataset", values="Macro-F1 % [95% CI]").loc[list(SCORE_ORDER)]
    latex_table(tables / "quality_primary.tex", ["Method", "HAPT", "MotionSense"],
                [(NAMES[mode], row.HAPT, row.MotionSense) for mode, row in wide.iterrows()], "lrr")
    latex_table(tables / "firmware_cost.tex",
                ["Dataset", "Method", "Base (B)", "Cache (B)", "Edit (B)", "Erases", "Edit (insn)", "Boot (insn)"],
                systems_view.itertuples(index=False, name=None), "llrrrrrr")
    latex_table(tables / "recovery.tex", ["Dataset", "Campaign", "Method", "Boot", "Retry"],
                fault_view.itertuples(index=False, name=None), "lllrr")


def figure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GRID, "axes.axisbelow": True, "axes.facecolor": PAPER,
        "figure.facecolor": PAPER, "xtick.color": SUBTLE, "ytick.color": SUBTLE,
        "grid.color": GRID, "grid.linewidth": 0.75, "text.color": INK,
        "svg.fonttype": "none",
    })


def heading(fig, title, subtitle):
    fig.text(0.067, 0.955, title, fontsize=20, weight="bold", va="top")
    fig.text(0.067, 0.891, subtitle, fontsize=10, color=SUBTLE, va="top")


def save(fig, out, name):
    path = out / "figures"
    path.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(path / f"{name}.{extension}", dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)


def recovery_figure(faults, renode, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
    heading(fig, "Interrupted edits recover exactly",
            "Native C campaigns over both datasets  |  Cortex-M4 checks run separately in Renode")
    items = (("power_cut", "proposed"), ("compaction", "proposed"), ("duplicate", "proposed"),
             ("duplicate", "no_idempotency"), ("checkpoint_retry", "proposed"),
             ("checkpoint_retry", "no_checkpoint_receipt"), ("integrity", "proposed"),
             ("integrity", "no_crc_validation"))
    aggregate = faults.groupby(["campaign", "mode"])[["cases", "successful_retries"]].sum()
    labels = ["Interrupted edit", "Interrupted compaction", "Lost ACK · journal",
              "Lost ACK · no retry IDs", "Compacted · receipt", "Compacted · no receipt",
              "Payload bit flip · journal", "Payload bit flip · no CRC"]
    for i, item in enumerate(items):
        row = aggregate.loc[item]
        pct = 100 * row.successful_retries / row.cases
        left.barh(i, 100, color="#eef3f6", height=0.62, zorder=1)
        left.barh(i, pct, color=TEAL, height=0.62, zorder=2)
        left.text(103, i, f"{int(row.successful_retries):,}/{int(row.cases):,}",
                  va="center", fontsize=9, color=INK if pct else ORANGE)
    left.set_yticks(range(len(items)), labels)
    left.invert_yaxis()
    left.set_xlim(0, 127)
    left.set_xticks([0, 25, 50, 75, 100])
    left.set_xlabel("Correct outcome after retry (%)")
    left.set_title("A   Software fault campaigns", loc="left", pad=13)
    left.grid(axis="x")
    masks = ("before", "torn_low_half", "after")
    checks = renode[renode["case"] == "open_corrected"]["dataset"].nunique() * renode["seed"].nunique()
    for x in range(6):
        for y, mask in enumerate(masks):
            selected = renode[(renode["layout"] == "packed") &
                              (renode["case"] == f"edit_cut_{x}_{mask}_recovery")]
            committed = int(selected["last_id"].min()) > 0
            color = TEAL if committed else "#dce8f6"
            right.scatter(x, y, marker="s", s=540, color=color, edgecolors="white", linewidths=1.2)
            right.text(x, y, f"{len(selected)}/{checks}", ha="center", va="center",
                       fontsize=10, weight="bold", color="white" if committed else INK)
    right.set_xlim(-0.55, 5.55)
    right.set_ylim(2.5, -0.5)
    right.set_yticks(range(3), ["Before write", "Lower half written", "Word completed"])
    right.set_xticks(range(6), ["ID", "Range", "Class", "Epoch", "CRC", "Commit"])
    right.set_title("B   Exact state after Cortex-M4 reset", loc="left", pad=13)
    right.set_xlabel("Interrupted flash word")
    right.text(0, -0.26, "■ Pre-edit state", transform=right.transAxes, color=BLUE, fontsize=9)
    right.text(0.53, -0.26, "■ Committed edit", transform=right.transAxes, color=TEAL, fontsize=9)
    for ax in (left, right):
        ax.tick_params(axis="both", length=0)
    fig.subplots_adjust(left=0.17, right=0.966, top=0.75, bottom=0.24, wspace=0.47)
    save(fig, out, "fig1_recovery")


def effect_figure(quality, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.25), sharey=True)
    heading(fig, "Label correction recovers recognition accuracy",
            "Held-out participants  |  Two-second edits  |  Paired 95% participant-bootstrap intervals")
    modes = ("source_only", "add_only", "no_prior", "short_history", "coarse_blocks", "proposed")
    names = [NAMES[mode] for mode in modes]
    primary = quality[quality["edit_length_s"] == 2]
    visible = primary[primary["mode"].isin(modes)]
    limits = (min(-5, float(np.floor(visible.delta_ci_low.min())) - 1),
              max(5, float(np.ceil(visible.delta_ci_high.max())) + 1))
    for ax, dataset, letter in zip(axes, DATASETS, "AB", strict=True):
        frame = primary[primary["dataset"] == dataset].set_index("mode").loc[list(modes)]
        ax.axvline(0, color="#9eaebd", lw=1.0, linestyle="--")
        for i, (mode, row) in enumerate(frame.iterrows()):
            color = TEAL if mode == "proposed" else (ORANGE if mode == "source_only" else BLUE)
            ax.errorbar(row.delta_vs_uncorrected_pp, i,
                        xerr=[[row.delta_vs_uncorrected_pp - row.delta_ci_low],
                              [row.delta_ci_high - row.delta_vs_uncorrected_pp]],
                        color=color, fmt="o", capsize=3, lw=1.55, ms=7 if mode == "proposed" else 5)
        ax.set_title(f"{letter}   {dataset}  ·  n = {int(frame.participants.iloc[0])}", loc="left", pad=12)
        ax.set_xlabel("Macro-F1 gain over wrong labels (percentage points)")
        ax.grid(axis="x")
        ax.tick_params(length=0)
        ax.set_yticks(np.arange(len(modes)), names)
        ax.set_xlim(*limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.invert_yaxis()
    axes[0].set_ylim(len(modes) - 0.45, -0.55)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    fig.text(0.195, 0.061, "Full vectors, unpacked summaries and packed summaries agree in every run.",
             fontsize=9, color=SUBTLE)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.74, bottom=0.18, wspace=0.18)
    save(fig, out, "fig2_paired_effects")


def sensitivity_figure(quality, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    heading(fig, "Longer annotation errors leave more room for recovery",
            "Mean participant macro-F1  |  Three corruption seeds per participant")
    colors = {"no_correction": ORANGE, "add_only": BLUE, "proposed": TEAL, "source_only": SUBTLE}
    for ax, dataset, letter in zip(axes, DATASETS, "AB", strict=True):
        for mode in ("no_correction", "add_only", "source_only", "proposed"):
            frame = quality[(quality["dataset"] == dataset) & (quality["mode"] == mode)].sort_values("edit_length_s")
            x, y = frame["edit_length_s"].to_numpy(), frame["macro_f1_pct"].to_numpy()
            color = colors[mode]
            ax.plot(x, y, "o-" if mode != "source_only" else "--", color=color,
                    label=NAMES[mode], lw=2.2 if mode == "proposed" else 1.65)
            if mode == "proposed":
                ax.fill_between(x, frame["ci_low"], frame["ci_high"], color=TEAL, alpha=0.13)
        ax.set_title(f"{letter}   {dataset}", loc="left", pad=12)
        ax.set_xlabel("Mislabeled seconds per edited interval")
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.84, 3.16)
        ax.grid(axis="y")
        ax.tick_params(length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.52, 0.033), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.76, bottom=0.23, wspace=0.28)
    save(fig, out, "fig3_error_duration")


def resource_figure(systems, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.3), sharey=True)
    heading(fig, "Storage and edit costs of the same exact classifier",
            "Cortex-M4 in Renode  |  Median across two datasets and three seeds per dataset")
    order = tuple(SYSTEM_NAMES)
    frame = systems.groupby("layout").median(numeric_only=True).loc[list(order)]
    metrics = (
        ("calibration_payload_bytes", "A   Calibration", "Payload (bytes)"),
        ("edit_program_bytes", "B   Flash programming", "Bytes per clean edit"),
        ("correction_instructions_median", "C   CPU work", "Guest instructions (thousands)"),
    )
    for ax, (key, title, ylabel) in zip(axes, metrics, strict=True):
        for i, layout in enumerate(order):
            val = int(frame.loc[layout, key])
            v = val / 1000 if key.endswith("median") else val
            color = TEAL if layout == "packed" else (ORANGE if layout == "snapshot" else "#9aafd1")
            ax.barh(i, v, height=0.61, color=color)
            ax.annotate(f"{v:,.1f}" if key.endswith("median") else f"{val:,}", (v, i),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(range(len(order)), [SYSTEM_NAMES[layout] for layout in order])
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel(ylabel)
        ax.grid(axis="x")
        ax.tick_params(length=0)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.21)
    axes[0].invert_yaxis()
    fig.text(0.187, 0.052, "The snapshot erases a sector for every edit. Journal erases are charged at compaction.",
             fontsize=9, color=SUBTLE)
    fig.subplots_adjust(left=0.185, right=0.965, top=0.76, bottom=0.18, wspace=0.33)
    save(fig, out, "fig4_firmware_cost")


def confusion_figure(confusion, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    heading(fig, "Which activity confusions change after repair",
            "Change in row-normalized confusion  |  Mean over people and seeds  |  Percentage points")
    cmap = LinearSegmentedColormap.from_list("repair", [ORANGE, "#f8fafb", TEAL])
    matrices = []
    for dataset in DATASETS:
        g = confusion[confusion["dataset"] == dataset].copy()
        denom = g.groupby(["subject", "seed", "mode", "actual"])["count"].transform("sum")
        g["pct"] = 100 * g["count"] / denom
        means = g.groupby(["actual", "predicted", "mode"])["pct"].mean().unstack("mode")
        diff = (means["proposed"] - means["no_correction"]).unstack("predicted")
        matrices.append(diff.loc[list(LABELS[dataset]), list(LABELS[dataset])].to_numpy())
    bound = max(5.0, float(np.max(np.abs(matrices))))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    for ax, dataset, matrix, letter in zip(axes, DATASETS, matrices, "AB", strict=True):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                value = matrix[i, j]
                ax.text(j, i, "0.0" if abs(value) < 0.05 else f"{value:+.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > bound * 0.66 else INK, weight="bold" if i == j else "normal")
        ax.set_xticks(np.arange(6), LABELS[dataset], rotation=30, ha="right")
        ax.set_yticks(np.arange(6), LABELS[dataset])
        ax.set_title(f"{letter}   {dataset}", loc="left", pad=12)
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("Recorded activity")
        ax.tick_params(length=0)
    fig.subplots_adjust(left=0.11, right=0.88, top=0.75, bottom=0.20, wspace=0.30)
    cax = fig.add_axes([0.9, 0.26, 0.014, 0.43])
    fig.colorbar(im, cax=cax, ticks=[-bound, 0, bound], format="%.1f")
    save(fig, out, "fig5_confusion_change")


def write_cost_figure(systems, stress, out):
    """Use the measured transaction sizes and show the host stress point separately."""
    packed = systems[systems["layout"] == "packed"]
    snapshot = systems[systems["layout"] == "snapshot"]
    edit = float(packed["edit_program_bytes"].median())
    checkpoint = float(packed["checkpoint_program_bytes"].median())
    snapshot_edit = float(snapshot["edit_program_bytes"].median())
    stress_bytes = stress["completed_program_bytes"].sum() / stress["verified_edits"].sum()
    filled = np.array([1, 2, 4, 8, 16])
    fig, ax = plt.subplots(figsize=(3.45, 1.98))
    ax.set_xscale("log", base=2)
    ax.axhline(snapshot_edit, color=ORANGE, lw=1.1, ls="--")
    ax.plot(filled, edit + checkpoint / filled, marker="o", ms=3.5, color=TEAL, lw=1.2)
    ax.plot(8, stress_bytes, marker="D", ms=4, color=BLUE)
    ax.annotate(f"Snapshot {snapshot_edit:.0f}", (16, snapshot_edit), xytext=(-1, 5),
                textcoords="offset points", ha="right", fontsize=7.8, color=ORANGE)
    ax.annotate(f"Host stress {stress_bytes:.1f}", (8, stress_bytes), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=7.8, color=BLUE)
    ax.annotate(f"Clean journal {edit + checkpoint / 16:.2f}", (16, edit + checkpoint / 16),
                xytext=(0, -11), textcoords="offset points", ha="right", fontsize=7.8, color=TEAL)
    ax.set_xlim(0.91, 17.5)
    ax.set_ylim(0, 225)
    ax.set_xticks(filled, [str(k) for k in filled])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("Bytes per edit", fontsize=8.5)
    ax.set_xlabel("Journal slots", fontsize=8.5)
    ax.grid(axis="y", lw=0.45)
    ax.tick_params(length=0, labelsize=8)
    fig.subplots_adjust(left=0.187, right=0.98, top=0.965, bottom=0.23)
    save(fig, out, "fig6_journal_write_cost")


def run(out=RESULTS):
    out = Path(out)
    scores = pd.read_csv(out / "scores.csv")
    faults = fault_summary(pd.read_csv(out / "faults.csv"))
    data = pd.read_csv(out / "dataset_inventory.csv")
    renode = pd.read_csv(out / "renode.csv")
    stress = pd.read_csv(out / "stress.csv")
    people = participant_scores(scores)
    quality = quality_summary(people)
    systems = firmware_summary(renode, data)
    display_tables(out, quality, faults, systems, data, scores, stress)
    figure_style()
    recovery_figure(faults, renode, out)
    effect_figure(quality, out)
    sensitivity_figure(quality, out)
    resource_figure(systems, out)
    confusion_figure(pd.read_csv(out / "confusion.csv"), out)
    write_cost_figure(systems, stress, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS)
    run(parser.parse_args().out)
