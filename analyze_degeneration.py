#!/usr/bin/env python3
"""
analyze_degeneration.py - separate recognition quality from decoding pathology.

    python analyze_degeneration.py --dir bench_out_all200

WHY THIS EXISTS
---------------
Surya scored grapheme CER 2.91 on KHOB level 1. Taken at face value that says
it cannot read Khmer, and inspection says the opposite: on clean lines it is
frequently EXACT where our own model is off by a character.

    gt     ឧបនាយករដ្ឋមន្ត្រី
    ours   ឧបនាយករជ្ឋមន្ត្រី     (ជ្ឋ for ដ្ឋ)
    surya  ឧបនាយករដ្ឋមន្ត្រី     exact

What the 2.91 actually measures is autoregressive degeneration: on hard input
the model falls into a repetition loop and emits the same phrase dozens of
times, occasionally drifting into Lao script. One 38-character reference drew
a 685-character prediction.

Reporting 2.91 as "Surya's accuracy" would be exactly the error we caught four
times in our own harness - a plausible number that measures something other
than what it claims - except that this time the error flatters us. That is
precisely when it needs catching.

WHAT IT REPORTS
---------------
Three numbers per engine, not one:

    raw CER          as measured, pathology included
    collapsed CER    after consecutive repeated substrings are reduced to one
                     occurrence - an estimate of recognition quality alone
    degeneration     the share of lines the collapse actually changed, and how
                     far over-length the output ran

The collapse is a single regex applied identically to every engine, so it
cannot favour one. For a CTC recogniser it is very nearly a no-op, which is
itself the point: CTC cannot emit more characters than it has timesteps, so it
has no way to run away. The comparison of the two columns is the finding.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from khmer_metrics import edit_distance, graphemes  # noqa: E402

# Any substring of >= MIN_RUN characters repeated back to back collapses to one
# copy. Non-greedy so the SHORTEST repeating unit wins, which is what a
# degeneration loop looks like.
MIN_RUN = 6
_REPEAT = re.compile(r"(.{%d,}?)\1+" % MIN_RUN, re.DOTALL)

# Khmer is U+1780..U+17FF; Lao U+0E80..U+0EFF; Thai U+0E00..U+0E7F
_KH = re.compile(r"[ក-៿]")
_LAO = re.compile(r"[຀-໿]")
_THAI = re.compile(r"[฀-๿]")


def collapse(s: str, rounds: int = 6) -> str:
    """Reduce consecutive repetitions of any substring to a single occurrence."""
    for _ in range(rounds):
        out = _REPEAT.sub(r"\1", s)
        if out == s:
            return s
        s = out
    return s


def cer(preds, gts) -> float:
    d = n = 0
    for p, g in zip(preds, gts):
        gg = graphemes(g)
        d += edit_distance(graphemes(p), gg)
        n += len(gg)
    return d / max(n, 1)


def load(path: Path):
    preds, gts = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in ln:
            continue
        p, g = ln.split("\t", 1)
        preds.append(p)
        gts.append(g)
    return preds, gts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a bench_out_* directory")
    ap.add_argument("--show", type=int, default=3,
                    help="worst degenerate lines to print per engine")
    ap.add_argument("--ratio", type=float, default=1.5,
                    help="pred/gt length ratio counted as over-generation")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("pred_*.txt"))
    if not files:
        raise SystemExit(f"no pred_*.txt in {args.dir}")

    print(f"{'engine':<12} {'raw CER':>9} {'collapsed':>10} {'gain':>8} "
          f"{'looped':>8} {'over-len':>9} {'max x':>7}")
    print("-" * 70)

    worst = {}
    for f in files:
        name = f.stem.replace("pred_", "")
        preds, gts = load(f)
        if not preds:
            continue
        col = [collapse(p) for p in preds]

        raw, cc = cer(preds, gts), cer(col, gts)
        looped = sum(1 for a, b in zip(preds, col) if a != b)
        ratios = [len(p) / max(len(g), 1) for p, g in zip(preds, gts)]
        over = sum(1 for r in ratios if r > args.ratio)
        gain = (raw - cc) / max(raw, 1e-9)

        print(f"{name:<12} {raw:9.4f} {cc:10.4f} {gain:+7.1%} "
              f"{looped:7d}  {over:8d} {max(ratios):7.1f}")

        worst[name] = sorted(
            ((len(p) / max(len(g), 1), p, g, c)
             for p, g, c in zip(preds, gts, col)),
            reverse=True)[:args.show]

    print(f"\n{len(load(files[0])[0])} lines per engine.  "
          f"'looped' = lines the collapse changed;  'over-len' = lines longer "
          f"than {args.ratio}x the reference;  'max x' = worst length ratio.")

    # ---- per-line distribution ----
    # An aggregate CER is a ratio of total edits to total reference length, so
    # a handful of 100x over-generations dominate it completely. The median
    # line is immune to that tail, and the gap between the two columns is the
    # honest summary: how the engine does usually, versus how badly it can
    # fail. Reporting only the aggregate would understate Surya as badly as
    # reporting only the median would flatter it.
    print("\nper-line grapheme CER distribution")
    print(f"{'engine':<12} {'median':>9} {'p75':>9} {'p90':>9} {'p99':>9} "
          f"{'worst':>9}")
    print("-" * 70)
    for f in files:
        name = f.stem.replace("pred_", "")
        preds, gts = load(f)
        if not preds:
            continue
        per = []
        for p, g in zip(preds, gts):
            gg = graphemes(g)
            per.append(edit_distance(graphemes(p), gg) / max(len(gg), 1))
        per.sort()
        n = len(per)

        def q(frac):
            return per[min(n - 1, int(frac * n))]

        print(f"{name:<12} {q(0.50):9.4f} {q(0.75):9.4f} {q(0.90):9.4f} "
              f"{q(0.99):9.4f} {per[-1]:9.2f}")

    # ---- script drift ----
    print("\nscript drift (non-Khmer characters emitted for Khmer references)")
    print("-" * 70)
    for f in files:
        name = f.stem.replace("pred_", "")
        preds, gts = load(f)
        lao = thai = 0
        for p, g in zip(preds, gts):
            if not _KH.search(g):
                continue                      # reference is not Khmer; skip
            if _LAO.search(p):
                lao += 1
            elif _THAI.search(p):
                thai += 1
        if lao or thai:
            print(f"{name:<12} Lao {lao:4d}   Thai {thai:4d}")
        else:
            print(f"{name:<12} none")

    # ---- examples ----
    for name, rows in worst.items():
        if not rows or rows[0][0] < args.ratio:
            continue
        print(f"\n\nworst over-generation - {name}")
        for ratio, p, g, c in rows:
            if ratio < args.ratio:
                continue
            print(f"\n  x{ratio:.0f}  ({len(g)} -> {len(p)} chars, "
                  f"{len(c)} after collapse)")
            print(f"  gt        {g[:110]}")
            print(f"  pred      {p[:110]}...")
            print(f"  collapsed {c[:110]}")


if __name__ == "__main__":
    sys.exit(main())
