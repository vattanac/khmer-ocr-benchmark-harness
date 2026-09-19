#!/usr/bin/env python3
"""
khmer_metrics.py - cluster-aware error rates for Khmer text recognition.

Khmer writes a syllable as a base consonant plus optional COENG subscripts and
vowel signs, all rendered as one visual unit. Scoring per Unicode code point
therefore flatters a recogniser: one wrongly placed diacritic can be counted as
several independent errors, and two systems that differ only in how they order
the code points inside a cluster can look far apart.

This module scores over orthographic clusters instead, following KHCWER
(https://github.com/keosaly/KHCWER):

    CER = (S + D + I) / N

where S, D and I are substituted, deleted and inserted clusters under
Levenshtein alignment and N is the number of clusters in the reference.

Report `cer_grapheme`. `cer_codepoint` is provided only for comparison with
work that scores per code point; the two are not interchangeable.
"""

from __future__ import annotations

import re

# base consonant or independent vowel, then any COENG + subscript pairs,
# then any combining marks; anything else falls through as a single unit
_GRAPHEME = re.compile(
    r"[ក-អឥ-ឳៜ](?:្[ក-ឳ])*"
    r"[឴-៑៝๎]*|."
)


def graphemes(s: str) -> list[str]:
    """Split into approximate Khmer grapheme clusters: base + COENG + marks."""
    return _GRAPHEME.findall(s)


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences (of clusters or characters)."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(preds: list[str], gts: list[str]) -> dict:
    """Corpus-level scores over aligned prediction/reference lists."""
    cd = cn = gd = gn = exact = 0
    for p, g in zip(preds, gts):
        cd += edit_distance(list(p), list(g)); cn += len(g)
        pg, gg = graphemes(p), graphemes(g)
        gd += edit_distance(pg, gg); gn += len(gg)
        exact += (p == g)
    return {
        "cer_codepoint": cd / max(cn, 1),
        "cer_grapheme": gd / max(gn, 1),
        "line_accuracy": exact / max(len(gts), 1),
        "n": len(gts),
    }


if __name__ == "__main__":
    # tiny self-check: a dropped subscript is ONE cluster error, not several
    ref, hyp = "បេក្ខជន", "បេកូជន"
    print("reference clusters:", graphemes(ref))
    print("hypothesis clusters:", graphemes(hyp))
    s = score([hyp], [ref])
    print(f"grapheme CER {s['cer_grapheme']:.3f}  codepoint CER {s['cer_codepoint']:.3f}")
