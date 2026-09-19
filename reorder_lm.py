#!/usr/bin/env python3
"""
reorder_lm.py - LM-arbitrated repair of Khmer character-order errors.

    # as a library (what bench_ocr.py --lm-reorder does)
    from reorder_lm import lm_reorder
    fixed = lm_reorder(text, lm, margin=1.0)

    # quick sanity check against the LM on a few known cases
    python reorder_lm.py check --lm khmer_lm.pkl.gz

    # repair a prediction file (pred<TAB>label) and report CER before/after
    python reorder_lm.py eval --lm khmer_lm.pkl.gz --pred bench_out/pred_ours-lm.txt

THE PROBLEM
-----------
CTC reads strictly left to right in TIME, but Khmer does not render in logical
order: in `ប្រ` the `្រ` hook sits LEFT of the `ប` it attaches to, and the
pre-posed vowels េ ែ ៃ ោ ៅ render entirely left of their consonant. So the
recogniser sometimes emits the visual order - `្របកាស` for `ប្រកាស` - which is
invalid Khmer.

WHY A PLAIN REGEX FAILED (measured: 0.1996 -> 0.2027 on KHOB level-1)
---------------------------------------------------------------------
Two different faults produce the same surface anomaly, and a regex cannot tell
them apart:

    `្របកាស`   base present, misordered   -> swapping is correct
    `្រកាស`    base absent entirely       -> swapping corrupts a good consonant

THE FIX
-------
Arbitrate with the character LM. At every anomaly site build both candidate
strings - as-emitted and swapped - score the FULL line with the n-gram model,
and keep the swap only when it beats the original by `margin` log-units. When
the base is genuinely missing, the swapped string is even less like Khmer than
the original, the LM scores it worse, and nothing changes. Safe by
construction: the only lines it can touch are ones the LM is confident about.

Applied to every engine's output equally - it is Unicode repair, not a thumb on
our scale.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COENG = "្"
_CONS = "ក-អ"                       # ក-អ consonants
_PRE_VOWELS = "េែៃោៅ"  # េ ែ ៃ ោ ៅ render left of base

# a COENG cluster (one or more ្C) with NO consonant before it, followed by a
# consonant that could be its base
_ORPHAN_COENG = re.compile(
    f"(?<![{_CONS}])((?:{COENG}[{_CONS}])+)([{_CONS}])")

# a pre-posed vowel with no consonant/COENG-cluster before it, followed by a
# consonant (and its subscripts) it could belong to
_ORPHAN_VOWEL = re.compile(
    f"(?<![{_CONS}])([{_PRE_VOWELS}])([{_CONS}](?:{COENG}[{_CONS}])*)")


def line_logp(lm, s: str) -> float:
    """Total LM log-probability of a line, start-of-line marker included."""
    t = "\x02" + s
    total = 0.0
    for i in range(1, len(t)):
        total += lm.logp(t[max(0, i - (lm.order - 1)):i], t[i])
    return total


def _candidates(s: str):
    """Yield (start, end, replacement) for every repairable anomaly in s."""
    for m in _ORPHAN_COENG.finditer(s):
        #  ្រ + ប  ->  ប + ្រ    (move the whole coeng cluster after the base)
        yield m.start(), m.end(), m.group(2) + m.group(1)
    for m in _ORPHAN_VOWEL.finditer(s):
        #  េ + ត្រ  ->  ត្រ + េ  (vowel after the consonant cluster)
        yield m.start(), m.end(), m.group(2) + m.group(1)


def lm_reorder(s: str, lm, margin: float = 1.0) -> str:
    """
    Repair character-order anomalies in s, keeping a swap only when the LM
    scores the swapped line at least `margin` log-units better.

    Sites are handled left to right; each accepted swap can expose a new site
    (a fixed cluster may reveal an orphan vowel), so the scan repeats until
    nothing changes.
    """
    for _ in range(4):                       # more than enough in practice
        base_score = None
        changed = False
        for start, end, rep in sorted(_candidates(s)):
            cand = s[:start] + rep + s[end:]
            if base_score is None:
                base_score = line_logp(lm, s)
            cs = line_logp(lm, cand)
            if cs > base_score + margin:
                s, base_score, changed = cand, cs, True
                break                        # offsets moved; rescan
        if not changed:
            return s
    return s


# ----------------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------------

def cmd_check(args) -> None:
    from khmer_ocr import CharLM
    lm = CharLM.load(args.lm)
    cases = [
        ("្របកាស", "misordered base  -> should swap"),
        ("្រកាស", "missing base     -> should NOT swap"),
        ("េសចក្តី", "misordered vowel -> should swap"),
        ("ប្រកាសព័ត៌មាន", "already correct  -> must not change"),
    ]
    print(f"LM order {lm.order}, {len(lm.ctx):,} contexts, margin {args.margin}\n")
    for s, why in cases:
        out = lm_reorder(s, lm, args.margin)
        mark = "CHANGED " if out != s else "kept    "
        print(f"{mark} {s!r:>22} -> {out!r:<22} ({why})")


def cmd_eval(args) -> None:
    from khmer_ocr import CharLM
    from khmer_metrics import edit_distance, graphemes
    lm = CharLM.load(args.lm)

    pairs = []
    for ln in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if "\t" in ln:
            p, g = ln.split("\t", 1)
            pairs.append((p, g))
    print(f"{len(pairs):,} lines from {args.pred}")

    def cer(preds):
        d = n = 0
        for p, g in zip(preds, (g for _, g in pairs)):
            gg = graphemes(g)
            d += edit_distance(graphemes(p), gg)
            n += len(gg)
        return d / max(n, 1)

    before = [p for p, _ in pairs]
    after = [lm_reorder(p, lm, args.margin) for p in before]
    nch = sum(1 for a, b in zip(before, after) if a != b)
    cb, ca = cer(before), cer(after)
    print(f"changed  : {nch} line(s)")
    print(f"CER      : {cb:.4f} -> {ca:.4f}  ({(ca-cb)/max(cb,1e-9):+.1%})")
    if args.show:
        shown = 0
        for (p, g), a in zip(pairs, after):
            if a != p and shown < args.show:
                print(f"\n  gt     {g}\n  before {p}\n  after  {a}")
                shown += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="sanity-check on known cases")
    c.add_argument("--lm", required=True)
    c.add_argument("--margin", type=float, default=1.0)
    c.set_defaults(fn=cmd_check)

    e = sub.add_parser("eval", help="repair a pred<TAB>label file, report CER")
    e.add_argument("--lm", required=True)
    e.add_argument("--pred", required=True)
    e.add_argument("--margin", type=float, default=1.0)
    e.add_argument("--show", type=int, default=5)
    e.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
