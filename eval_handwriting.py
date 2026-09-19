#!/usr/bin/env python3
"""
eval_handwriting.py - establish the honest starting line on real Khmer handwriting.

    python eval_handwriting.py --split khmer_handwritten/testset.parquet --ckpt best_v2.pt
    python eval_handwriting.py --split khmer_handwritten/testset.parquet --ckpt best_v2.pt \
        --lm khmer_lm.pkl.gz

This runs the CURRENT document-trained model on the real handwriting test set.
Expect a poor number - the model has never seen handwriting. That is the point:
this is the baseline we improve from in Phase 1 (KCC + Transformer + fine-tuning
on this very data). You cannot claim to reach #1 without first knowing exactly
where you start, measured the same way the field measures.

Metric: grapheme CER over Khmer character clusters (the same cluster-aware idea
as the official KHCWER metric), plus exact-line rate.

Dataset: SoyVitou/khmer-handwritten-dataset-4.2k - columns 'image' (PNG bytes)
and 'text' (Khmer line label).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from khmer_ocr import build_model, pick_device  # noqa: E402
from khmer_metrics import edit_distance, graphemes  # noqa: E402
from bench_ocr import clean  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="a .parquet split with image/text")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=12, help="sample predictions to print")
    ap.add_argument("--out", default="handwriting_preds.tsv")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    df = pd.read_parquet(args.split)
    if args.limit:
        df = df.iloc[:args.limit]
    print(f"handwriting test lines: {len(df)}")

    dev = pick_device(args.device)
    st = torch.load(args.ckpt, map_location="cpu")
    itos = st["vocab"]["itos"] if "vocab" in st else st["itos"]
    model = build_model(len(itos) + 1, args.height).to(dev)
    model.load_state_dict(st["model"])
    model.eval()

    lm = None
    if args.lm:
        from khmer_ocr import CharLM
        lm = CharLM.load(args.lm)
    from khmer_ocr import beam_decode
    from khmer_ocr import greedy_decode
    print(f"device {dev}, {len(itos)} classes, decoder = "
          f"{'beam+LM' if lm else 'greedy'}\n")

    def load_img(cell):
        b = cell["bytes"] if isinstance(cell, dict) else cell
        return Image.open(io.BytesIO(b)).convert("L")

    imgs = [load_img(c) for c in df["image"].tolist()]
    gts = [clean(str(t)) for t in df["text"].tolist()]

    preds = []
    for i in range(0, len(imgs), args.batch):
        chunk = imgs[i:i + args.batch]
        tens, widths = [], []
        for im in chunk:
            w = max(8, int(round(im.width * args.height / im.height)))
            w = min(w, args.max_width)
            a = np.asarray(im.resize((w, args.height)), dtype="float32") / 255.0
            tens.append(torch.from_numpy((a - 0.5) / 0.5).unsqueeze(0))
            widths.append(w)
        mw = int(np.ceil(max(widths) / 8) * 8)
        xs = torch.ones(len(chunk), 1, args.height, mw)
        for j, tt in enumerate(tens):
            xs[j, :, :, :tt.shape[-1]] = tt
        with torch.no_grad():
            logits = model(xs.to(dev))
        if lm is None:
            preds.extend(greedy_decode(logits, itos))
        else:
            lp = logits.log_softmax(-1).cpu()
            for j, w in enumerate(widths):
                T = max(1, w // 8)
                preds.append(beam_decode(lp[j, :T].tolist(), itos, lm,
                                         args.beam, args.alpha, args.beta))
    preds = [clean(p) for p in preds]

    d = sum(edit_distance(graphemes(p), graphemes(g)) for p, g in zip(preds, gts))
    n = sum(len(graphemes(g)) for g in gts)
    exact = sum(1 for p, g in zip(preds, gts) if p == g)
    cer = d / max(n, 1)

    print(f"{'='*54}")
    print(f"grapheme CER (KHCWER-style) : {cer:.4f}   ({cer*100:.1f}%)")
    print(f"exact-line accuracy         : {100*exact/len(gts):.1f}%")
    print(f"{'='*54}")
    print(f"\nSoTA reference (KTRWS, 2026) handwriting CER: ~8.9%")
    print(f"This is the document-trained model with no handwriting exposure -")
    print(f"the number that Phase 1 (fine-tuning on this data) improves from.\n")

    # a few examples, sorted worst-first, so the failure modes are visible
    rows = sorted(zip(gts, preds),
                  key=lambda gp: -edit_distance(graphemes(gp[1]), graphemes(gp[0]))
                  / max(len(graphemes(gp[0])), 1))
    print("sample (worst first):")
    for g, p in rows[:args.show]:
        print(f"\n  gt   {g}\n  pred {p}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("pred\tgt\n")
        for p, g in zip(preds, gts):
            f.write(f"{p}\t{g}\n")
    print(f"\nall predictions -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
