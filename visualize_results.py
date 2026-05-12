#!/usr/bin/env python3
"""
Build visualizations of all train.py runs for team review.

Inputs:
  runs/<timestamp>/results.json    — final test/val accuracy
  runs/<timestamp>/full_run.log    — per-epoch train/val accuracy + confusion matrices

Outputs (analysis/):
  fig1_main_results.png        — headline bar chart, 9-user run, with chance line
  fig2_scale_comparison.png    — BVP/BAP/BVP+BAP across 3 scales (README 4u, ours 1u, ours 9u)
  fig3_training_curves.png     — per-epoch train+val curves for each mode in the 9-user run
  fig4_confusion_matrices.png  — 3 heatmaps side-by-side for the 9-user run
  summary_results.csv          — machine-readable table

Usage: python visualize_results.py
"""
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHANCE_6 = 1.0 / 6  # 0.1667 for 6-class

RUNS_ROOT = Path("runs")
OUT = Path("analysis")
OUT.mkdir(exist_ok=True)

# ---------- README baseline (user10–13, hardcoded from csi-auth/README.md) ----------
README_BASELINE = {
    "label": "csi-auth README\n(4 users, 3K samples)",
    "BVP": 0.433,
    "BAP": 0.490,
    "BVP+BAP": 0.463,
    "n_samples": 3000,
}

MODE_ORDER = ["BVP", "BAP", "BVP+BAP"]
MODE_COLORS = {"BVP": "#377eb8", "BAP": "#e41a1c", "BVP+BAP": "#4daf4a"}


# ---------- log parsing ----------
EPOCH_LINE = re.compile(
    r"accuracy:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*val_accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)"
)
MODE_HEADER = re.compile(r"^\s+(BVP\+BAP|BVP|BAP)\s*$")
CM_HEADER = "Confusion matrix (normalized):"


def parse_log(log_path):
    """Returns dict: mode -> {trajectory: [(train_acc, val_acc), ...], cm: 6x6 ndarray or None}."""
    text = log_path.read_text(errors="ignore")
    # Split by mode headers (lines like "  BVP" inside the ====== block)
    out = {}
    current_mode = None
    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")

    # First pass: split text into chunks per mode by finding mode header lines
    lines = text.splitlines()
    mode_starts = []
    for i, line in enumerate(lines):
        m = MODE_HEADER.match(line)
        if m:
            mode_starts.append((i, m.group(1)))

    # Append sentinel
    mode_starts.append((len(lines), None))

    for k in range(len(mode_starts) - 1):
        start, mode = mode_starts[k]
        end, _ = mode_starts[k + 1]
        if mode is None:
            continue
        chunk = "\n".join(lines[start:end])

        # Trajectory: take the LAST epoch summary in each Epoch N/30 block.
        # Easier: split chunk by "Epoch N/30" markers, find one EPOCH_LINE in each.
        traj = []
        epoch_blocks = re.split(r"Epoch \d+/\d+", chunk)
        for block in epoch_blocks[1:]:  # first split is before Epoch 1
            matches = EPOCH_LINE.findall(block)
            if matches:
                acc, loss, vacc, vloss = matches[-1]  # the final line in block has full epoch summary
                traj.append((float(acc), float(vacc), float(loss), float(vloss)))

        # Confusion matrix
        cm = None
        cm_idx = chunk.find(CM_HEADER)
        if cm_idx >= 0:
            after = chunk[cm_idx + len(CM_HEADER):]
            # Read up to 6 lines containing matrix rows
            mat_rows = []
            for line in after.splitlines():
                line = line.strip().lstrip("[").rstrip("]")
                nums = re.findall(r"-?[0-9]*\.?[0-9]+", line)
                if len(nums) == 6:
                    mat_rows.append([float(x) for x in nums])
                    if len(mat_rows) == 6:
                        break
            if len(mat_rows) == 6:
                cm = np.array(mat_rows)

        out[mode] = {"trajectory": traj, "cm": cm}

    return out


def load_run(run_dir):
    """Returns dict with results + per-mode trajectories/cms."""
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    log_path = run_dir / "full_run.log"
    parsed = parse_log(log_path) if log_path.exists() else {}
    return results, parsed


# ---------- collect all runs ----------
all_runs = {}
for d in sorted(RUNS_ROOT.iterdir()):
    if not (d / "results.json").exists():
        continue
    results, parsed = load_run(d)
    all_runs[d.name] = (results, parsed)
    print(f"loaded {d.name}: users={results['users']}  modes={[r['name'] for r in results['results']]}")

# Identify the canonical 9-user "all modes" run
nine_user = next(
    ((k, r, p) for k, (r, p) in all_runs.items()
     if r["users"] == "all" and len(r["results"]) == 3),
    None,
)
single_bvp = next(
    ((k, r, p) for k, (r, p) in all_runs.items()
     if r["users"] == ["user1"] and r["results"][0]["name"] == "BVP"),
    None,
)
single_bap = next(
    ((k, r, p) for k, (r, p) in all_runs.items()
     if r["users"] == ["user1"] and r["results"][0]["name"] == "BAP"),
    None,
)


# ---------- summary CSV ----------
csv_rows = [["run_id", "setting", "n_samples", "mode", "test_acc", "train_acc", "val_acc", "train_time_s"]]
for k, (r, _) in all_runs.items():
    setting = "user1-only" if r["users"] == ["user1"] else (
        "all 9 users" if r["users"] == "all" else str(r["users"])
    )
    for res in r["results"]:
        csv_rows.append([
            k, setting, r["n_samples"], res["name"],
            f"{res['accuracy']:.4f}",
            f"{res.get('final_train_acc', 0):.4f}",
            f"{res.get('final_val_acc', 0):.4f}",
            f"{res['train_time_s']:.1f}",
        ])
# Add README baseline
for mode in MODE_ORDER:
    csv_rows.append([
        "README_baseline", "4 users (user10-13)", README_BASELINE["n_samples"],
        mode, f"{README_BASELINE[mode]:.4f}", "", "", "",
    ])
with open(OUT / "summary_results.csv", "w", newline="") as f:
    csv.writer(f).writerows(csv_rows)
print(f"\nwrote {OUT/'summary_results.csv'}")


# ---------- Fig 1: headline 9-user results ----------
if nine_user is None:
    print("WARNING: no 9-user 'all modes' run found, skipping figure 1")
else:
    _, results, _ = nine_user
    test_acc = {r["name"]: r["accuracy"] for r in results["results"]}
    train_acc = {r["name"]: r.get("final_train_acc", 0) for r in results["results"]}
    val_acc = {r["name"]: r.get("final_val_acc", 0) for r in results["results"]}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(MODE_ORDER))
    w = 0.27
    bars1 = ax.bar(x - w, [train_acc[m] for m in MODE_ORDER], w, label="train", color="#cccccc")
    bars2 = ax.bar(x,     [val_acc[m]   for m in MODE_ORDER], w, label="val",   color="#888888")
    bars3 = ax.bar(x + w, [test_acc[m]  for m in MODE_ORDER], w,
                   label="test", color=[MODE_COLORS[m] for m in MODE_ORDER])
    for b, v in zip(bars3, [test_acc[m] for m in MODE_ORDER]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.1%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(CHANCE_6, color="red", linestyle="--", linewidth=1.2,
               label=f"chance ({CHANCE_6:.1%})")
    ax.set_xticks(x); ax.set_xticklabels(MODE_ORDER, fontsize=11)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, max(0.45, max(test_acc.values()) + 0.1))
    ax.set_title(f"9-user run — {results['n_samples']} samples, 6 gestures, "
                 f"{results['epochs']} epochs (full random 90/10 split)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_main_results.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/'fig1_main_results.png'}")


# ---------- Fig 2: scale comparison ----------
fig, ax = plt.subplots(figsize=(10, 5.5))
groups = []
if README_BASELINE:
    groups.append((README_BASELINE["label"], {m: README_BASELINE[m] for m in MODE_ORDER}))
if single_bvp and single_bap:
    _, r_bvp, _ = single_bvp
    _, r_bap, _ = single_bap
    groups.append((f"single user (user1)\n({r_bvp['n_samples']} samples)",
                   {"BVP": r_bvp["results"][0]["accuracy"],
                    "BAP": r_bap["results"][0]["accuracy"],
                    "BVP+BAP": None}))
if nine_user:
    _, r9, _ = nine_user
    groups.append((f"all 9 users\n({r9['n_samples']} samples)",
                   {res["name"]: res["accuracy"] for res in r9["results"]}))

x = np.arange(len(groups))
w = 0.27
for i, mode in enumerate(MODE_ORDER):
    ys = [g[1].get(mode) for g in groups]
    bars = ax.bar(x + (i - 1) * w,
                  [y if y is not None else 0 for y in ys],
                  w, label=mode, color=MODE_COLORS[mode])
    for j, (b, y) in enumerate(zip(bars, ys)):
        if y is None:
            ax.text(b.get_x() + b.get_width()/2, 0.005, "—",
                    ha="center", fontsize=11, color="#777777")
        else:
            ax.text(b.get_x() + b.get_width()/2, y + 0.008, f"{y:.1%}",
                    ha="center", fontsize=9)
ax.axhline(CHANCE_6, color="red", linestyle="--", linewidth=1.2,
           label=f"chance ({CHANCE_6:.1%})")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 0.55)
ax.set_title("Test accuracy across data scales — BAP holds, BVP collapses with diversity")
ax.legend(loc="upper right")
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig2_scale_comparison.png", dpi=130)
plt.close(fig)
print(f"wrote {OUT/'fig2_scale_comparison.png'}")


# ---------- Fig 3: training curves (9-user run) ----------
if nine_user and any(p["trajectory"] for p in nine_user[2].values()):
    _, _, parsed = nine_user
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, mode in zip(axs, MODE_ORDER):
        traj = parsed.get(mode, {}).get("trajectory", [])
        if not traj:
            ax.set_title(f"{mode} (no trajectory)"); continue
        epochs = np.arange(1, len(traj) + 1)
        train = [t[0] for t in traj]
        val   = [t[1] for t in traj]
        ax.plot(epochs, train, "o-", label="train", color=MODE_COLORS[mode],
                markersize=4, linewidth=1.5)
        ax.plot(epochs, val, "s--", label="val", color=MODE_COLORS[mode],
                markersize=4, linewidth=1.5, alpha=0.6)
        ax.axhline(CHANCE_6, color="red", linestyle=":", linewidth=1.0,
                   label=f"chance ({CHANCE_6:.1%})")
        ax.set_xlabel("epoch")
        ax.set_title(f"{mode}  (final test={[r for r in nine_user[1]['results'] if r['name']==mode][0]['accuracy']:.1%})")
        ax.set_ylim(0.10, 0.45)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=9)
    axs[0].set_ylabel("accuracy")
    fig.suptitle("9-user run — training trajectories show BVP plateau vs BAP / BVP+BAP learning",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_training_curves.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/'fig3_training_curves.png'}")


# ---------- Fig 4: confusion matrices (9-user run) ----------
if nine_user:
    _, results, parsed = nine_user
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, mode in zip(axs, MODE_ORDER):
        cm = parsed.get(mode, {}).get("cm")
        if cm is None:
            ax.set_title(f"{mode} (no CM in log)"); ax.axis("off"); continue
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(6):
            for j in range(6):
                v = cm[i, j]
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_xticks(range(6)); ax.set_xticklabels([f"m{i+1}" for i in range(6)])
        ax.set_yticks(range(6)); ax.set_yticklabels([f"m{i+1}" for i in range(6)])
        ax.set_xlabel("predicted")
        if mode == "BVP":
            ax.set_ylabel("true")
        acc = [r for r in results["results"] if r["name"] == mode][0]["accuracy"]
        diag = float(np.trace(cm) / cm.sum() if cm.sum() else 0.0)
        ax.set_title(f"{mode}  test={acc:.1%}  diag={diag:.1%}", fontsize=11)
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02, label="row-normalized prob")
    fig.suptitle("9-user run confusion matrices — BVP collapses to predicting class 1; "
                 "BAP/BVP+BAP show partial diagonal", fontsize=12)
    fig.savefig(OUT / "fig4_confusion_matrices.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig4_confusion_matrices.png'}")


# ---------- print compact summary ----------
print("\n" + "=" * 78)
print(f"{'setting':28s}  {'n':>6s}  {'BVP':>8s}  {'BAP':>8s}  {'BVP+BAP':>8s}")
print("-" * 78)
print(f"{'README baseline (u10-13)':28s}  {README_BASELINE['n_samples']:>6d}  "
      f"{README_BASELINE['BVP']:>7.1%}  {README_BASELINE['BAP']:>7.1%}  {README_BASELINE['BVP+BAP']:>7.1%}")
if single_bvp and single_bap:
    _, r_bvp, _ = single_bvp; _, r_bap, _ = single_bap
    print(f"{'single user (user1)':28s}  {r_bvp['n_samples']:>6d}  "
          f"{r_bvp['results'][0]['accuracy']:>7.1%}  {r_bap['results'][0]['accuracy']:>7.1%}  "
          f"{'—':>8s}")
if nine_user:
    _, r9, _ = nine_user
    accs = {res['name']: res['accuracy'] for res in r9['results']}
    print(f"{'all 9 users':28s}  {r9['n_samples']:>6d}  "
          f"{accs.get('BVP', 0):>7.1%}  {accs.get('BAP', 0):>7.1%}  {accs.get('BVP+BAP', 0):>7.1%}")
print("=" * 78)
print(f"\nDone. Artifacts in {OUT}/")
