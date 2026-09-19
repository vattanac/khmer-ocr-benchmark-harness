#!/usr/bin/env python3
"""
verify_tables.py - recompute Tables I and II of the paper from the released
per-line predictions.

    python verify_tables.py

Needs nothing but Python: no models, no images, no benchmark data. It reads
predictions/level{1,2}/pred_<engine>.txt, where each line is

    <prediction>\\t<reference>

and rescores them with the same cluster-aware metric used in the paper
(khmer_metrics.py). If the printed numbers match the "paper" column, the
reported results follow from the released outputs.

To check the scoring itself rather than the outputs, look at khmer_metrics.py:
it is about forty lines. To regenerate the predictions from images, see
bench_ocr.py, which needs the KHOB dataset and the OCR engines installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from khmer_metrics import score  # noqa: E402

HERE = Path(__file__).resolve().parent

# engine -> (grapheme CER, exact-line accuracy) as printed in the paper
PAPER = {
    "level1": {
        "title": "Table I  Document regions (KHOB level 1; 1,862 regions)",
        "rows": [("ours-lm", 0.1692, 0.287), ("ours", 0.1978, 0.262),
                 ("surya", 0.3167, 0.203), ("tess-best", 0.3941, 0.149),
                 ("tesseract", 0.4318, 0.145)],
    },
    "level2": {
        "title": "Table II  Scene-text regions (KHOB level 2; 2,215 regions)",
        "rows": [("ours-lm", 0.2865, 0.469), ("ours", 0.3134, 0.423),
                 ("tess-best", 0.5414, 0.157), ("tesseract", 0.5464, 0.156),
                 ("surya", 0.6851, 0.355)],
    },
}
TOL = 0.0005          # 4-decimal agreement


def read(path: Path) -> tuple[list[str], list[str]]:
    preds, refs = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        preds.append(parts[0])
        refs.append(parts[1] if len(parts) > 1 else "")
    return preds, refs


def main() -> int:
    ok = True
    for level, spec in PAPER.items():
        print(f"\n{spec['title']}")
        print(f"  {'engine':14s}{'CER':>9s}{'paper':>9s}{'exact':>9s}{'paper':>9s}   ")
        for engine, cer_paper, acc_paper in spec["rows"]:
            f = HERE / "predictions" / level / f"pred_{engine}.txt"
            if not f.exists():
                print(f"  {engine:14s}  missing: {f}")
                ok = False
                continue
            preds, refs = read(f)
            s = score(preds, refs)
            cer_ok = abs(s["cer_grapheme"] - cer_paper) < TOL
            acc_ok = abs(s["line_accuracy"] - acc_paper) < 0.001
            ok = ok and cer_ok and acc_ok
            mark = "ok" if (cer_ok and acc_ok) else "MISMATCH"
            print(f"  {engine:14s}{s['cer_grapheme']:9.4f}{cer_paper:9.4f}"
                  f"{s['line_accuracy']:9.3f}{acc_paper:9.3f}   {mark}")

    print("\n" + ("All values reproduce the paper within 4 decimal places."
                  if ok else "Some values do not match. See the rows marked MISMATCH."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
