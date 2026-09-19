#!/usr/bin/env python3
"""
bench_ocr.py - run several OCR engines over the KHOB benchmark and compare them.

    # our model, greedy
    python bench_ocr.py --data ../bench/data/level-1 --ckpt best_v2.pt --engines ours

    # our model with the language model, plus Tesseract, side by side
    python bench_ocr.py --data ../bench/data/level-1 --ckpt best_v2.pt \\
        --lm khmer_lm.pkl.gz --alpha 0.35 --beta 1.5 --engines ours,ours-lm,tesseract

    # quick smoke test on 50 lines
    python bench_ocr.py --data ../bench/data/level-2 --ckpt best_v2.pt \\
        --engines ours,tesseract --limit 50

    # every competitor we can obtain
    python bench_ocr.py --data ../bench/data/level-1/khob-... --ckpt best_v2.pt \\
        --lm khmer_lm.pkl.gz --lm-reorder \\
        --engines ours-lm,tesseract,tesseract-best,surya \\
        --tessdata-best ~/tessdata_best

ENGINES
-------
    ours            this project's CRNN, greedy CTC decoding
    ours-lm         the same checkpoint with LM beam search
    tesseract       Tesseract, whichever khm.traineddata is installed
    tesseract-best  Tesseract against tessdata_best (see below)
    surya           Surya, datalab-to/surya - 90+ languages including Khmer
    hf:<repo-id>    any line-level Khmer recogniser on the HuggingFace hub

A NOTE ON FAIRNESS TO TESSERACT
-------------------------------
`brew install tesseract-lang` installs the *tessdata* build of khm.traineddata,
which is integer-quantised for speed. The *tessdata_best* repository ships the
full-precision LSTM, which Tesseract's own documentation recommends when
accuracy matters. Reporting a margin against only the fast model invites the
obvious objection, so `tesseract-best` exists and both should be reported.

WHAT IT DOES
------------
The KHOB dataset is LabelMe JSON: one file per page, each holding polygons whose
`label` is the Khmer text of that region. This walks those files, crops every
labelled region, and feeds the identical crop to each engine - so the comparison
is of recognisers, not of anyone's line segmentation.

It reports grapheme CER (a Khmer written character is base + COENG + subscript +
vowel sign - the honest unit), codepoint CER, and exact-line accuracy, broken
down per task folder. It also writes `pred<TAB>label` files in the format KHOB's
own scripts/evaluate.py expects, so the numbers can be cross-checked with their
implementation rather than only ours.

FAIRNESS NOTES
--------------
Both sides are NFC-normalised, and zero-width space (U+200B) is stripped from
both by default: Khmer uses it as an invisible word separator, engines disagree
about emitting it, and leaving it in measures annotation convention rather than
recognition. Pass --keep-zwsp to see the difference it makes.

Tesseract runs with --psm 7 (treat the image as a single text line), which is
the setting that matches how the crops are fed. Anything else would handicap it
unfairly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from khmer_ocr import build_model, pick_device  # noqa: E402
from khmer_metrics import edit_distance, graphemes  # noqa: E402

ZWSP = "​"
ZWNJ = "‌"
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------

def open_upright(path: Path):
    """
    Open an image the way its annotator saw it.

    Phone photos carry an EXIF orientation tag: the sensor writes landscape
    pixels and records "rotate this 90 degrees for display". PIL reports the raw
    sensor dimensions, while LabelMe annotates the displayed image - so on KHOB
    level-2, 751 of 800 photos have width and height transposed relative to
    their polygons. Without this, every crop is taken from a sideways page and
    the scores are meaningless (all three engines measured ~0.95 CER).
    """
    from PIL import Image, ImageOps
    return ImageOps.exif_transpose(Image.open(path))


def find_image(js: Path, image_path: str | None) -> Path | None:
    """
    Resolve a LabelMe imagePath. The KHOB files carry Windows separators and
    sometimes point one directory up, but the image is in practice the
    same-stemmed file beside the JSON - so try that first and fall back.
    """
    for ext in IMG_EXT:
        for cand in (js.with_suffix(ext), js.with_suffix(ext.upper())):
            if cand.exists():
                return cand
    if image_path:
        name = image_path.replace("\\", "/").split("/")[-1]
        for base in (js.parent, js.parent.parent):
            c = base / name
            if c.exists():
                return c
            for ext in IMG_EXT:
                c2 = (base / name).with_suffix(ext)
                if c2.exists():
                    return c2
    return None


import re

# Khmer consonants; COENG is U+17D2 and always binds to the consonant BEFORE it
_CONS = "ក-អ"
_COENG_FIRST = re.compile(f"(?<![{_CONS}])្([{_CONS}])([{_CONS}])")


def fix_coeng_order(s: str) -> str:
    """
    Repair a subscript emitted before its base consonant.

    CTC decodes strictly left to right in time, but Khmer does not render that
    way: in `ប្រ` the `្រ` hook sits slightly LEFT of the `ប` it hangs under, so
    the recogniser can emit `្រប` - visually first, logically second. The result
    is not merely wrong, it is invalid Khmer: a COENG cannot open a cluster.

    So where a COENG sequence appears with no base consonant before it, the base
    is the character that follows; swap them. Applied to every engine's output
    equally - it is Unicode repair, not a thumb on our scale.

    MEASURED: this HURTS slightly (0.1996 -> 0.2027 on KHOB level-1), so it is
    off by default. Two different faults produce a leading COENG and this rule
    cannot separate them:

        `្របកាស`  base present, misordered  -> swap is correct
        `្រកាស`   base absent entirely      -> swap corrupts a good consonant

    The principled version scores both orderings with the character LM and keeps
    the likelier one; that needs the LM available here, which Tesseract's path
    does not have. Left as future work rather than shipped as a fake win.
    """
    prev = None
    while prev != s:                 # repeat: one swap can expose another
        prev = s
        s = _COENG_FIRST.sub(lambda m: m.group(2) + "្" + m.group(1), s)
    return s


def clean(text: str, keep_zwsp: bool = False) -> str:
    # LabelMe stores literal backslash-t in some labels; both it and real tabs
    # must go, because the output format is tab-separated
    text = text.replace("\\t", " ").replace("\t", " ").replace("\\n", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    if not keep_zwsp:
        text = text.replace(ZWSP, "").replace(ZWNJ, "")
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()


def load_regions(data_dir: Path, limit: int | None, keep_zwsp: bool,
                 min_chars: int = 1):
    """Yield (task, image path, bbox, ground-truth text) for every labelled region."""
    from PIL import Image

    out, bad, scaled = [], 0, 0
    for js in sorted(data_dir.rglob("*.json")):
        # macOS writes AppleDouble "._name.json" stubs when unzipping; they are
        # not JSON and counting them as corrupt labels is alarming and wrong
        if js.name.startswith("._") or js.name.startswith("."):
            continue
        try:
            d = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            bad += 1
            continue
        if "shapes" not in d:
            continue
        img = find_image(js, d.get("imagePath"))
        if img is None:
            continue

        # KHOB task-4 stores polygons in the ORIGINAL page's coordinate space
        # (596x842) while shipping a smaller rendition (509x720). Cropping raw
        # coordinates then lands on the wrong line entirely - which looks like
        # catastrophic recognition failure and is nothing of the kind.
        sx = sy = 1.0
        try:
            aw, ah = open_upright(img).size        # as the annotator saw it
            jw, jh = d.get("imageWidth"), d.get("imageHeight")
            if jw and jh and (abs(aw - jw) > 2 or abs(ah - jh) > 2):
                sx, sy = aw / jw, ah / jh
                scaled += 1
        except Exception:
            pass
        # task name = the deepest directory that looks like task-N, else parent
        task = next((p.name for p in [js.parent, *js.parents] if p.name.startswith("task")),
                    js.parent.name)
        for sh in d["shapes"]:
            txt = clean(str(sh.get("label") or ""), keep_zwsp)
            pts = sh.get("points") or []
            if len(txt) < min_chars or len(pts) < 2:
                continue
            xs = [p[0] * sx for p in pts]
            ys = [p[1] * sy for p in pts]
            out.append((task, img, (min(xs), min(ys), max(xs), max(ys)), txt))

    if bad:
        print(f"note: {bad} malformed JSON file(s) skipped")
    if scaled:
        print(f"note: {scaled} page(s) had polygons rescaled to the shipped image size")
    if limit:
        # spread the sample across tasks rather than taking the first N
        step = max(1, len(out) // limit)
        out = out[::step][:limit]
    return out


def crop(img_cache, path: Path, box, pad_frac: float = 0.0, pad_min: int = 3):
    """
    Crop a region with padding PROPORTIONAL to its height.

    A fixed pixel pad clips the leading consonant on short regions - the model
    then emits `្របកាស` for `ប្រកាស`, having never seen the `ប`. Annotation
    polygons are drawn tight to the ink, so some margin is always needed, and
    how much depends on the text size.
    """
    im = img_cache.get(path)
    if im is None:
        im = open_upright(path).convert("L")
        img_cache.clear()          # pages are large; one at a time is plenty
        img_cache[path] = im
    x0, y0, x1, y1 = box
    pad = max(pad_min, int(round((y1 - y0) * pad_frac)))
    x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
    x1 = min(im.width, int(x1) + pad); y1 = min(im.height, int(y1) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return im.crop((x0, y0, x1, y1))


# ----------------------------------------------------------------------------
# engines
# ----------------------------------------------------------------------------

class OursEngine:
    """The project's CRNN, greedy or with LM beam search."""

    def __init__(self, ckpt, device, height=64, max_width=1600, batch=32,
                 lm=None, alpha=0.35, beta=1.5, beam=8):
        import torch
        self.torch = torch
        self.dev = pick_device(device)
        st = torch.load(ckpt, map_location="cpu")
        self.itos = st["vocab"]["itos"]
        self.model = build_model(len(self.itos) + 1, height).to(self.dev)
        self.model.load_state_dict(st["model"])
        self.model.eval()
        self.h, self.mw, self.batch = height, max_width, batch
        self.lm, self.alpha, self.beta, self.beam = lm, alpha, beta, beam
        self.name = "ours-lm" if lm else "ours"

    def run(self, crops):
        import numpy as np
        torch = self.torch
        from khmer_ocr import beam_decode
        from khmer_ocr import greedy_decode

        out = []
        for i in range(0, len(crops), self.batch):
            chunk = crops[i:i + self.batch]
            tens, widths = [], []
            for im in chunk:
                w = max(8, int(round(im.width * self.h / im.height)))
                w = min(w, self.mw)
                a = np.asarray(im.resize((w, self.h)), dtype="float32") / 255.0
                tens.append(torch.from_numpy((a - 0.5) / 0.5).unsqueeze(0))
                widths.append(w)
            mw = int(np.ceil(max(widths) / 8) * 8)
            xs = torch.ones(len(chunk), 1, self.h, mw)
            for j, tt in enumerate(tens):
                xs[j, :, :, :tt.shape[-1]] = tt
            with torch.no_grad():
                logits = self.model(xs.to(self.dev))
            if self.lm is None:
                out.extend(greedy_decode(logits, self.itos))
            else:
                lp = logits.log_softmax(-1).cpu()
                for j, w in enumerate(widths):
                    T = max(1, w // 8)
                    out.append(beam_decode(lp[j, :T].tolist(), self.itos, self.lm,
                                           self.beam, self.alpha, self.beta))
        return out


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang="khm", psm=7):
        self.lang, self.psm = lang, psm
        try:
            subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        except Exception:
            raise SystemExit("tesseract not found - brew install tesseract tesseract-lang")

    def run(self, crops):
        out = []
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "line.png")
            for im in crops:
                # Tesseract does better with a margin and a bit more height
                if im.height < 32:
                    im = im.resize((max(8, im.width * 32 // im.height), 32))
                im.save(p)
                try:
                    r = subprocess.run(
                        ["tesseract", p, "stdout", "-l", self.lang, "--psm", str(self.psm)],
                        capture_output=True, text=True, timeout=60)
                    out.append(clean(r.stdout))
                except Exception:
                    out.append("")
        return out


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------

def score_pairs(preds, gts):
    cd = cn = gd = gn = ex = 0
    for p, g in zip(preds, gts):
        cd += edit_distance(list(p), list(g)); cn += len(g)
        pg, gg = graphemes(p), graphemes(g)
        gd += edit_distance(pg, gg); gn += len(gg)
        ex += (p == g)
    n = max(len(gts), 1)
    return {"cer_g": gd / max(gn, 1), "cer_c": cd / max(cn, 1),
            "acc": ex / n, "n": len(gts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder of LabelMe JSON (searched recursively)")
    ap.add_argument("--ckpt", default=None, help="our checkpoint, for the 'ours' engines")
    ap.add_argument("--lm", default=None)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--engines", default="ours,tesseract",
                    help="comma list. Built in: ours, ours-lm, tesseract. "
                         "From engines_extra.py: tesseract-best, surya, "
                         "hf:<repo-id> (e.g. hf:Darayut/khmer-text-recognition)")
    ap.add_argument("--tessdata-best", default=None,
                    help="directory holding tessdata_best khm.traineddata, for "
                         "the tesseract-best engine. The default Homebrew "
                         "install ships the FAST model; benchmarking only "
                         "against that overstates our margin.")
    ap.add_argument("--tess-lang", default="khm",
                    help="traineddata to use for tesseract-best: khm (language) "
                         "or Khmer (script)")
    ap.add_argument("--hf-device", default="cpu",
                    help="device for hf: engines (cpu, mps, cuda)")
    ap.add_argument("--hf-trust", action="store_true",
                    help="allow trust_remote_code for hf: engines. This EXECUTES "
                         "Python from the model repository - read it first.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all regions")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-zwsp", action="store_true")
    ap.add_argument("--pad-frac", type=float, default=0.0,
                    help="extra crop padding as a fraction of region height; "
                         "0 keeps a flat 3px, which measured best - Khmer lines "
                         "sit close and a bigger margin drags in the next line")
    ap.add_argument("--coeng-fix", action="store_true",
                    help="repair subscripts emitted before their base. MEASURED "
                         "SLIGHTLY WORSE (0.1996 -> 0.2027) because it cannot "
                         "distinguish a misordered base from a missing one, so "
                         "it is off by default. See fix_coeng_order.")
    ap.add_argument("--lm-reorder", action="store_true",
                    help="LM-arbitrated repair of character-order errors "
                         "(needs --lm). Unlike --coeng-fix it scores both "
                         "orderings with the character LM and only swaps when "
                         "the LM is confident, so a missing base is left "
                         "alone. Applied to every engine's output equally.")
    ap.add_argument("--reorder-margin", type=float, default=1.0,
                    help="log-prob margin a swap must win by (higher = more "
                         "conservative)")
    ap.add_argument("--out", default="bench_out", help="directory for prediction files")
    ap.add_argument("--dump-worst", type=int, default=15)
    args = ap.parse_args()

    regions = load_regions(Path(args.data), args.limit or None, args.keep_zwsp)
    if not regions:
        raise SystemExit(f"no labelled regions found under {args.data}")
    tasks = sorted({t for t, _, _, _ in regions})
    print(f"regions : {len(regions):,}  across {len(tasks)} task(s): {', '.join(tasks)}")

    print("cropping...", flush=True)
    cache, crops, keep = {}, [], []
    for rec in regions:
        c = crop(cache, rec[1], rec[2], args.pad_frac)
        if c is not None:
            crops.append(c); keep.append(rec)
    gts = [r[3] for r in keep]
    print(f"crops   : {len(crops):,}\n")

    wanted = [e.strip() for e in args.engines.split(",") if e.strip()]
    engines = []
    lm = None
    if "ours-lm" in wanted or args.lm_reorder:
        if not args.lm:
            raise SystemExit("--lm is required for ours-lm and --lm-reorder")
        from khmer_ocr import CharLM
        lm = CharLM.load(args.lm)
    for w in wanted:
        if w == "ours":
            engines.append(OursEngine(args.ckpt, args.device, args.height,
                                      args.max_width, args.batch))
        elif w == "ours-lm":
            engines.append(OursEngine(args.ckpt, args.device, args.height,
                                      args.max_width, args.batch,
                                      lm=lm, alpha=args.alpha, beta=args.beta,
                                      beam=args.beam))
        elif w == "tesseract":
            engines.append(TesseractEngine())
        else:
            from engines_extra import build as build_extra
            # A competitor that will not load must not cost us the engines that
            # will: report it and carry on, so one missing package does not
            # discard an hour of other engines' results.
            try:
                e = build_extra(w, args)
            except SystemExit as err:
                print(f"SKIPPING {w}: {err}\n")
                continue
            if e is None:
                raise SystemExit(
                    f"unknown engine: {w}\n"
                    f"  known: ours, ours-lm, tesseract, tesseract-best, "
                    f"surya, hf:<repo-id>")
            engines.append(e)
    if not engines:
        raise SystemExit("no engines could be constructed")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for eng in engines:
        t0 = time.time()
        print(f"running {eng.name} ...", flush=True)
        try:
            preds = eng.run(crops)
        except SystemExit as err:
            print(f"  FAILED: {err}\n")
            continue
        except Exception as err:
            print(f"  FAILED: {type(err).__name__}: {err}\n")
            continue
        dt = time.time() - t0
        # a short return would silently misalign every prediction with the
        # wrong reference and produce a plausible, meaningless score
        if len(preds) != len(crops):
            print(f"  FAILED: returned {len(preds)} predictions for "
                  f"{len(crops)} crops - dropping this engine\n")
            continue
        if args.coeng_fix:
            fixed = sum(1 for p in preds if fix_coeng_order(p) != p)
            preds = [fix_coeng_order(p) for p in preds]
            if fixed:
                print(f"  COENG order repaired on {fixed} line(s)")
        if args.lm_reorder:
            from reorder_lm import lm_reorder
            new = [lm_reorder(p, lm, args.reorder_margin) for p in preds]
            fixed = sum(1 for a, b in zip(preds, new) if a != b)
            preds = new
            print(f"  LM reorder changed {fixed} line(s)")
        results[eng.name] = preds
        with (outdir / f"pred_{eng.name}.txt").open("w", encoding="utf-8") as f:
            for p, g in zip(preds, gts):
                f.write(f"{p}\t{g}\n")
        print(f"  {dt:.0f}s  ({len(crops)/max(dt,1e-9):.1f} lines/s)")

    # ---- overall ----
    print(f"\n{'engine':<12} {'graph CER':>10} {'cp CER':>9} {'line acc':>9} {'n':>7}")
    print("-" * 52)
    for name, preds in results.items():
        m = score_pairs(preds, gts)
        print(f"{name:<12} {m['cer_g']:10.4f} {m['cer_c']:9.4f} "
              f"{m['acc']:9.3f} {m['n']:7,}")

    # ---- per task ----
    if len(tasks) > 1:
        print(f"\nper task (grapheme CER)")
        print(f"{'task':<12} " + " ".join(f"{n:>12}" for n in results))
        print("-" * (13 + 13 * len(results)))
        for t in tasks:
            idx = [i for i, r in enumerate(keep) if r[0] == t]
            row = []
            for name, preds in results.items():
                m = score_pairs([preds[i] for i in idx], [gts[i] for i in idx])
                row.append(f"{m['cer_g']:12.4f}")
            print(f"{t:<12} " + " ".join(row) + f"   (n={len(idx)})")

    # ---- where we lose ----
    if args.dump_worst and len(results) > 1:
        names = list(results)
        a, b = names[0], names[-1]
        diffs = []
        for i, g in enumerate(gts):
            ca = edit_distance(graphemes(results[a][i]), graphemes(g)) / max(len(graphemes(g)), 1)
            cb = edit_distance(graphemes(results[b][i]), graphemes(g)) / max(len(graphemes(g)), 1)
            diffs.append((cb - ca, i))
        # diff = (b's error) - (a's error): positive means a did better
        diffs.sort(reverse=True)
        half = max(1, args.dump_worst // 2)
        print(f"\nlines where {a} beats {b} most:")
        for _, i in diffs[:half]:
            print(f"\n  gt   {gts[i]}\n  {a:<9}{results[a][i]}\n  {b:<9}{results[b][i]}")
        print(f"\nlines where {b} beats {a} most:")
        for _, i in diffs[-half:]:
            print(f"\n  gt   {gts[i]}\n  {a:<9}{results[a][i]}\n  {b:<9}{results[b][i]}")

    print(f"\nprediction files in {outdir}/  (pred<TAB>label, for KHOB evaluate.py)")


if __name__ == "__main__":
    sys.exit(main())
