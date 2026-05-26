#!/usr/bin/env python3
"""Generate a single-file HTML report from an ``experiments.py`` run directory.

Reads the JSON / PNG outputs and writes ``report.html`` alongside them.
Confusion matrices are rendered as color-coded tables (green for diagonal,
red for off-diagonal). Split PNGs are inlined as base64 so the HTML is
fully portable.

Usage:
    python generate_report.py                            # latest runs/exp_* dir
    python generate_report.py runs/exp_20260526_115314   # specific dir
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────
# Renderers
# ──────────────────────────────────────────────────────────
def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def cm_table(cm, classes, *, normalize=True) -> str:
    """Render a confusion matrix as a color-coded HTML table."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        rows = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    else:
        pct = cm

    n = len(classes)
    parts = ['<table class="cm">']
    parts.append("<tr><th></th>" + "".join(f"<th>{c}</th>" for c in classes) + "</tr>")
    for i, c in enumerate(classes):
        parts.append(f"<tr><th>{c}</th>")
        for j in range(n):
            v = float(pct[i, j])
            # Diagonal: green saturation by value; off-diagonal: red saturation by value
            intensity = max(0, min(255, int(255 - 230 * v)))
            if i == j:
                bg = f"rgb({intensity}, 255, {intensity})"
            else:
                bg = f"rgb(255, {intensity}, {intensity})"
            parts.append(f'<td style="background:{bg}">{v:.2f}</td>')
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def cm_pair(bvp_r, bap_r, title: str) -> str:
    """Render BVP + BAP conf matrices side by side with a card wrapper."""
    classes = bvp_r["classes"]
    out = ['<div class="matrix-card">']
    out.append(f"<h4>{title}</h4>")
    out.append('<div class="matrix-pair">')
    out.append("<div>")
    out.append(f"<div class=\"label\">BVP &mdash; test {bvp_r['accuracy']:.4f}</div>")
    out.append(cm_table(bvp_r["confusion_matrix"], classes))
    out.append("</div>")
    if bap_r is not None:
        out.append("<div>")
        out.append(f"<div class=\"label\">BAP &mdash; test {bap_r['accuracy']:.4f}</div>")
        out.append(cm_table(bap_r["confusion_matrix"], classes))
        out.append("</div>")
    out.append("</div></div>")
    return "".join(out)


def aggregate_cm(matrices):
    return np.sum(np.array(matrices), axis=0)


def render_table(rows, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="results"><tr>{head}</tr>{body}</table>'


# ──────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────
def section_config1(c1, run_dir):
    parts = ["<h2>Config 1 &mdash; Aggregate user classification</h2>"]
    parts.append("<p>Predict user (3 classes), motions 1&ndash;6 mixed. "
                 "Split: by-key 3-way, val_frac=0.1, test_frac=0.1. Chance = 33.3%.</p>")
    rows = [(r["mode"].upper(), f"{r['accuracy']:.4f}", f"{r['final_train_acc']:.4f}",
             f"{r['final_val_acc']:.4f}" if r["final_val_acc"] is not None else "N/A",
             r["n_train"], r.get("n_val", 0), r["n_test"], f"{r['train_time_s']}s")
            for r in c1]
    parts.append(render_table(rows, ["Mode", "Test", "Train", "Val", "n_train", "n_val", "n_test", "Time"]))

    bvp = next(r for r in c1 if r["mode"] == "bvp")
    bap = next(r for r in c1 if r["mode"] == "bap")
    parts.append("<h3>Confusion matrices (row-normalized)</h3>")
    parts.append(cm_pair(bvp, bap, "All motions mixed"))

    img = run_dir / "config1_splits.png"
    if img.exists():
        parts.append("<h3>Split composition</h3>")
        parts.append(f'<img class="figure" src="data:image/png;base64,{b64_image(img)}">')
    return "".join(parts)


def section_config2(c2, run_dir):
    parts = ["<h2>Config 2 &mdash; Per-motion user classification</h2>"]
    parts.append("<p>Predict user (3 classes), one experiment per motion. "
                 "Split: by-key 3-way, val_frac=0.1, test_frac=0.1. Chance = 33.3%.</p>")
    rows = []
    for r in c2:
        rows.append((r["motion"], r["mode"].upper(),
                     f"{r['accuracy']:.4f}",
                     f"{r['final_val_acc']:.4f}" if r["final_val_acc"] is not None else "N/A",
                     r["n_train"], r.get("n_val", 0), r["n_test"]))
    parts.append(render_table(rows, ["Motion", "Mode", "Test", "Val", "n_train", "n_val", "n_test"]))

    parts.append("<h3>Confusion matrices per motion</h3>")
    parts.append('<div class="cm-grid">')
    for m in sorted({r["motion"] for r in c2}):
        bvp = next((r for r in c2 if r["motion"] == m and r["mode"] == "bvp"), None)
        bap = next((r for r in c2 if r["motion"] == m and r["mode"] == "bap"), None)
        if bvp:
            parts.append(cm_pair(bvp, bap, f"Motion {m}"))
    parts.append("</div>")

    img = run_dir / "config2_splits.png"
    if img.exists():
        parts.append("<h3>Split composition (pooled across 6 motions)</h3>")
        parts.append(f'<img class="figure" src="data:image/png;base64,{b64_image(img)}">')
    return "".join(parts)


def section_config3(c3, run_dir):
    bvp_all = [r["accuracy"] for r in c3 if r["mode"] == "bvp"]
    bap_all = [r["accuracy"] for r in c3 if r["mode"] == "bap"]

    parts = ["<h2>Config 3 &mdash; Per-cell user classification (ideal settings)</h2>"]
    parts.append("<p>Predict user (3 classes), one experiment per "
                 "(motion, orientation, location) cell. Split: random 60/20/20. Chance = 33.3%.</p>")
    parts.append('<div class="summary-box">')
    parts.append(f"<b>Overall mean across {len(bvp_all)} cells:</b><br>")
    parts.append(f"&bull; BVP: <b>{np.mean(bvp_all):.4f}</b> (std {np.std(bvp_all):.4f})<br>")
    parts.append(f"&bull; BAP: <b>{np.mean(bap_all):.4f}</b> (std {np.std(bap_all):.4f})")
    parts.append("</div>")

    # Per-motion mean
    parts.append("<h3>Per-motion mean accuracy</h3>")
    rows = []
    for m in sorted({r["motion"] for r in c3}):
        bv = [r["accuracy"] for r in c3 if r["motion"] == m and r["mode"] == "bvp"]
        ba = [r["accuracy"] for r in c3 if r["motion"] == m and r["mode"] == "bap"]
        rows.append((m, f"{np.mean(bv):.4f}", f"{np.std(bv):.4f}",
                     f"{np.mean(ba):.4f}", f"{np.std(ba):.4f}", len(bv)))
    parts.append(render_table(rows, ["Motion", "BVP mean", "BVP std", "BAP mean", "BAP std", "Cells"]))

    # Aggregated conf matrices per motion
    parts.append("<h3>Aggregated confusion matrix per motion (summed across cells)</h3>")
    parts.append('<div class="cm-grid">')
    for m in sorted({r["motion"] for r in c3}):
        for mode in ("bvp", "bap"):
            cells = [r for r in c3 if r["motion"] == m and r["mode"] == mode]
            if not cells:
                continue
            agg = aggregate_cm([r["confusion_matrix"] for r in cells])
            fake = {"confusion_matrix": agg, "classes": cells[0]["classes"],
                    "accuracy": np.mean([r["accuracy"] for r in cells])}
            other = None  # already handled by mode loop
            # Render one matrix card per (motion, mode)
            parts.append('<div class="matrix-card">')
            parts.append(f"<h4>Motion {m} &mdash; {mode.upper()} (pooled {len(cells)} cells, mean {fake['accuracy']:.4f})</h4>")
            parts.append(cm_table(agg, cells[0]["classes"]))
            parts.append("</div>")
    parts.append("</div>")

    # Sample 5 spread cells per motion
    parts.append("<details><summary>Sample 5 spread cells per motion (worst, 25th, median, 75th, best by BVP)</summary>")
    for m in sorted({r["motion"] for r in c3}):
        bvp_cells = sorted([r for r in c3 if r["motion"] == m and r["mode"] == "bvp"],
                           key=lambda r: r["accuracy"])
        if not bvp_cells:
            continue
        n = len(bvp_cells)
        picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        parts.append(f"<h3>Motion {m}</h3>")
        parts.append('<div class="cm-grid">')
        for i in picks:
            rb = bvp_cells[i]
            ra = next((r for r in c3 if r["mode"] == "bap"
                       and r["motion"] == rb["motion"]
                       and r["orientation"] == rb["orientation"]
                       and r["location"] == rb["location"]), None)
            parts.append(cm_pair(rb, ra, f"ori={rb['orientation']}, loc={rb['location']} (n_test={rb['n_test']})"))
        parts.append("</div>")
    parts.append("</details>")

    # Full table of all 150 cells (sortable)
    parts.append("<details><summary>Full table: all cells (click to expand)</summary>")
    rows = []
    by_cell = {}
    for r in c3:
        key = (r["motion"], r["orientation"], r["location"])
        by_cell.setdefault(key, {})[r["mode"]] = r
    for key in sorted(by_cell):
        m, o, l = key
        bv = by_cell[key].get("bvp")
        ba = by_cell[key].get("bap")
        rows.append((m, o, l,
                     f"{bv['accuracy']:.4f}" if bv else "N/A",
                     f"{ba['accuracy']:.4f}" if ba else "N/A",
                     bv["n_train"] if bv else "?",
                     bv.get("n_val", 0) if bv else "?",
                     bv["n_test"] if bv else "?"))
    parts.append(render_table(rows, ["Motion", "Orientation", "Location",
                                     "BVP test", "BAP test", "n_train", "n_val", "n_test"]))
    parts.append("</details>")

    img = run_dir / "config3_splits.png"
    if img.exists():
        parts.append("<h3>Split composition (pooled across 150 cells)</h3>")
        parts.append(f'<img class="figure" src="data:image/png;base64,{b64_image(img)}">')
    return "".join(parts)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1200px; margin: 20px auto; padding: 0 24px; color: #222; line-height: 1.5; }
h1 { color: #1a4480; border-bottom: 2px solid #1a4480; padding-bottom: 8px; }
h2 { color: #1a4480; margin-top: 36px; border-bottom: 1px solid #c8d4e6; padding-bottom: 4px; }
h3 { color: #2a5b9e; margin-top: 24px; }
h4 { margin: 0 0 6px 0; color: #444; font-size: 14px; }
.meta { color: #666; font-size: 13px; }
.summary-box { background: #eef3fa; padding: 12px 16px; border-left: 4px solid #1a4480;
               margin: 12px 0; border-radius: 0 4px 4px 0; }
table.results { border-collapse: collapse; margin: 10px 0; font-size: 14px; }
table.results th, table.results td { border: 1px solid #ccc; padding: 6px 12px; text-align: right; }
table.results th { background: #f0f4fa; }
table.cm { border-collapse: collapse; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
table.cm th, table.cm td { border: 1px solid #888; padding: 4px 8px; text-align: center; min-width: 50px; }
table.cm th { background: #eee; font-weight: 600; }
.matrix-pair { display: flex; gap: 16px; flex-wrap: wrap; }
.matrix-card { border: 1px solid #ccc; padding: 10px 14px; border-radius: 4px; background: #fafafa;
               margin: 8px 0; display: inline-block; vertical-align: top; }
.cm-grid { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0; }
.label { font-size: 12px; color: #555; margin-bottom: 4px; font-family: ui-monospace, monospace; }
img.figure { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }
details { margin: 12px 0; border: 1px solid #ddd; border-radius: 4px; padding: 0 12px; background: #f9f9f9; }
details summary { cursor: pointer; font-weight: 600; padding: 10px 0; color: #1a4480; }
details[open] summary { border-bottom: 1px solid #ddd; margin-bottom: 8px; }
nav a { margin-right: 16px; color: #1a4480; text-decoration: none; font-weight: 600; }
nav a:hover { text-decoration: underline; }
.legend { font-size: 12px; color: #666; margin: 4px 0 12px 0; }
.legend .swatch { display: inline-block; width: 14px; height: 14px; vertical-align: middle;
                  border: 1px solid #888; margin-right: 4px; }
"""


def build_html(run_dir: Path, manifest_size=None):
    c1 = json.loads((run_dir / "config1_user_aggregate.json").read_text()) \
        if (run_dir / "config1_user_aggregate.json").exists() else None
    c2 = json.loads((run_dir / "config2_per_motion.json").read_text()) \
        if (run_dir / "config2_per_motion.json").exists() else None
    c3 = json.loads((run_dir / "config3_per_cell.json").read_text()) \
        if (run_dir / "config3_per_cell.json").exists() else None

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>Experiment Report &mdash; {run_dir.name}</title>")
    parts.append(f"<style>{CSS}</style>")
    parts.append("</head><body>")

    parts.append("<h1>BVP vs BAP &mdash; Experiment Report</h1>")
    parts.append(f"<p class='meta'>Run: <code>{run_dir.name}</code> &middot; "
                 f"Generated: {datetime.now().isoformat(timespec='seconds')} &middot; "
                 f"Users 1&ndash;3, gestures 1&ndash;6, keep-all dedup</p>")

    parts.append("<nav>")
    if c1: parts.append('<a href="#c1">Config 1</a>')
    if c2: parts.append('<a href="#c2">Config 2</a>')
    if c3: parts.append('<a href="#c3">Config 3</a>')
    parts.append("</nav>")

    parts.append('<div class="legend">'
                 '<span class="swatch" style="background:rgb(50,255,50)"></span>diagonal (correct prediction) '
                 '<span class="swatch" style="background:rgb(255,50,50)"></span>off-diagonal (confusion) '
                 '&mdash; intensity scales with value (0.00 white, 1.00 saturated)</div>')

    if c1:
        parts.append('<a id="c1"></a>')
        parts.append(section_config1(c1, run_dir))
    if c2:
        parts.append('<a id="c2"></a>')
        parts.append(section_config2(c2, run_dir))
    if c3:
        parts.append('<a id="c3"></a>')
        parts.append(section_config3(c3, run_dir))

    parts.append("</body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="Run directory (default: most recent runs/exp_* dir)")
    ap.add_argument("--out", default="report.html", help="Output filename (written inside run_dir)")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        candidates = sorted(Path("runs").glob("exp_*"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            sys.exit("no runs/exp_* directories found")
        run_dir = candidates[-1]

    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    html = build_html(run_dir)
    out_path = run_dir / args.out
    out_path.write_text(html)
    print(f"wrote {out_path}  ({len(html) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
