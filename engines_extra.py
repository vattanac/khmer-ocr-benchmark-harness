#!/usr/bin/env python3
"""
engines_extra.py - additional OCR engines for bench_ocr.py.

Adds three competitors beyond the default Tesseract build:

    tesseract-best      Tesseract with the tessdata_best LSTM model
    surya               Surya (datalab-to/surya), 90+ languages incl. Khmer
    hf:<repo-id>        any line-level Khmer recogniser on HuggingFace

WHY A SEPARATE FILE
-------------------
Each of these depends on a package that may not be installed, and each has an
API that has changed across releases. Keeping them here means bench_ocr.py
still runs with nothing but PyTorch and Tesseract present, and a broken or
missing competitor degrades to a clear error for that engine alone rather than
taking the whole benchmark down.

ON DEFENSIVE IMPORTS
--------------------
The Surya and HuggingFace wrappers try several call signatures in turn. This
is deliberate, not sloppiness: both projects have reorganised their public API
between minor versions, and a benchmark harness that only works against one
pinned release is a harness that will not be re-runnable in six months when
the thesis is examined.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

_TAG = re.compile(r"<[^>]+>")


# ----------------------------------------------------------------------------
# Tesseract with an alternate model directory
# ----------------------------------------------------------------------------

class TesseractBestEngine:
    """
    Tesseract driven against a chosen tessdata directory.

    This matters for fairness. `brew install tesseract-lang` installs the
    *tessdata* build of khm.traineddata: integer-quantised for speed. The
    *tessdata_best* repository ships the full-precision LSTM model, which is
    the one Tesseract's own documentation recommends when accuracy is the
    objective. Benchmarking against the fast model and reporting the margin
    invites the obvious objection, so we measure both and report both.

    Tesseract also ships a script-level `Khmer.traineddata` distinct from the
    language-level `khm.traineddata`; pass --tess-lang to select it.
    """

    def __init__(self, tessdata_dir: str, lang: str = "khm", psm: int = 7,
                 name: str = "tess-best"):
        self.dir, self.lang, self.psm, self.name = tessdata_dir, lang, psm, name
        if not os.path.isdir(tessdata_dir):
            raise SystemExit(f"tessdata directory not found: {tessdata_dir}")
        got = [f for f in os.listdir(tessdata_dir) if f.endswith(".traineddata")]
        if not any(f.startswith(lang) or f.lower().startswith(lang.lower()) for f in got):
            raise SystemExit(
                f"no {lang}.traineddata in {tessdata_dir}\n"
                f"  found: {', '.join(sorted(got)) or '(nothing)'}")

    def run(self, crops):
        from bench_ocr import clean
        out = []
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "line.png")
            for im in crops:
                if im.height < 32:
                    im = im.resize((max(8, im.width * 32 // im.height), 32))
                im.save(p)
                try:
                    r = subprocess.run(
                        ["tesseract", p, "stdout", "-l", self.lang,
                         "--psm", str(self.psm), "--tessdata-dir", self.dir],
                        capture_output=True, text=True, timeout=60)
                    out.append(clean(r.stdout))
                except Exception:
                    out.append("")
        return out


# ----------------------------------------------------------------------------
# Surya
# ----------------------------------------------------------------------------

def _surya_text(result) -> str:
    """Pull plain text out of whatever shape this Surya version returns."""
    # newest: result.blocks[].html
    blocks = getattr(result, "blocks", None)
    if blocks:
        parts = []
        for bl in blocks:
            h = getattr(bl, "html", None) or getattr(bl, "text", None) or ""
            parts.append(_TAG.sub(" ", h))
        return " ".join(parts)
    # older: result.text_lines[].text
    lines = getattr(result, "text_lines", None)
    if lines:
        return " ".join(getattr(ln, "text", "") or "" for ln in lines)
    # last resort
    return str(getattr(result, "text", "") or "")


class SuryaEngine:
    """
    Surya, with a synthetic whole-crop layout so its recogniser must read.

    MEASURED, not assumed: run in its default full-page mode over KHOB line
    crops, Surya returned nothing for 68% of them - its layout detector
    declares a 31-pixel-tall strip empty (in one case it labelled a Khmer
    title as a photograph of a person in a suit). Recognition never ran, so
    the benchmark was measuring the layout stage, not the reading.

    The fix is to TELL it where the text is: one LayoutBox covering the whole
    crop, label 'Text'. This is the same assumption every other engine already
    receives - Tesseract via --psm 7, our own model by construction - so it is
    fairness, not favouritism. With it, a line the default mode returned empty
    is read essentially perfectly:

        gt       ប្រកាសឱ្យប្រើច្បាប់ស្តីពីការគ្រប់គ្រងរដ្ឋបាលរាជធានី ...
        default  ''
        forced   ប្រកាសឱ្យប្រើច្បាប់ស្ដីពីការគ្រប់គ្រងរដ្ឋបាលរាជធានី ...

    The schema (surya-ocr with the llama.cpp backend, Sep 2026): LayoutBox
    requires polygon, label, raw_label, position, count; LayoutResult takes
    bboxes + image_bbox. Both are passed positionally to the recogniser with
    full_page=False. Older installs without these classes fall back to the
    previous behaviour.
    """

    name = "surya"

    def __init__(self, langs=("km",), batch: int = 16):
        self.langs, self.batch = list(langs), batch
        self.mode = None
        try:
            from surya.recognition import RecognitionPredictor
        except ImportError:
            raise SystemExit(
                "surya not installed.  pip install surya-ocr\n"
                "  (first run downloads model weights; needs network)")

        try:
            from surya.layout import LayoutBox, LayoutResult
            self._LayoutBox, self._LayoutResult = LayoutBox, LayoutResult
        except Exception:
            self._LayoutBox = self._LayoutResult = None

        # newest API: a shared inference manager
        try:
            from surya.inference import SuryaInferenceManager
            self.rec = RecognitionPredictor(SuryaInferenceManager())
            self.mode = "manager"
            return
        except Exception:
            pass

        # older API: predictors constructed directly, detection passed in
        self.rec = RecognitionPredictor()
        try:
            from surya.detection import DetectionPredictor
            self.det = DetectionPredictor()
        except Exception:
            self.det = None
        self.mode = "classic"

    def _layout_for(self, im):
        w, h = float(im.width), float(im.height)
        box = self._LayoutBox(
            polygon=[[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
            label="Text", raw_label="Text", position=0, count=1,
            confidence=1.0)
        return self._LayoutResult(bboxes=[box], image_bbox=[0.0, 0.0, w, h])

    def _call(self, images):
        if self.mode == "manager":
            if self._LayoutBox is not None:
                layouts = [self._layout_for(im) for im in images]
                try:
                    return self.rec(images, layouts, full_page=False)
                except TypeError:
                    return self.rec(images, layouts)
            return self.rec(images)
        # classic signature changed too: (images, langs, det) then (images, det)
        for args in ((images, [self.langs] * len(images), self.det),
                     (images, self.det),
                     (images,)):
            try:
                return self.rec(*args)
            except TypeError:
                continue
        raise SystemExit("could not find a working Surya call signature; "
                         "check the installed surya-ocr version")

    def run(self, crops):
        from bench_ocr import clean
        out = []
        fails = 0
        for i in range(0, len(crops), self.batch):
            chunk = [c.convert("RGB") for c in crops[i:i + self.batch]]
            try:
                res = self._call(chunk)
            except SystemExit:
                raise
            except Exception as e:
                # loud, per batch - a silent [""] here once fabricated a
                # confident 1.0000 CER out of an uninstalled binary
                fails += 1
                print(f"  surya batch {i // self.batch} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                out.extend([""] * len(chunk))
                continue
            for r in res:
                out.append(clean(_surya_text(r)))
        if fails:
            print(f"  surya: {fails} batch(es) failed and scored as empty")
        return out


# ----------------------------------------------------------------------------
# Any HuggingFace line recogniser
# ----------------------------------------------------------------------------

class HFEngine:
    """
    A line-level recogniser from the HuggingFace hub.

    Two shapes cover nearly every Khmer model published there:

      1. VisionEncoderDecoder / TrOCR - a processor turns the crop into pixel
         values, the model generates token ids, the processor decodes them.
      2. A custom architecture loaded with trust_remote_code, exposing its own
         predict() or generate() method.

    We try (1) then (2). trust_remote_code executes code from the model repo,
    so it is opt-in via --hf-trust and the repo should be read before enabling
    it - this is running a stranger's Python, not merely loading weights.
    """

    def __init__(self, repo: str, device="cpu", batch: int = 8, trust: bool = False):
        self.repo, self.batch, self.trust = repo, batch, trust
        self.name = "hf:" + repo.split("/")[-1][:18]
        self.kind = None
        import torch
        self.torch = torch
        self.dev = device

        try:
            from transformers import (AutoProcessor, TrOCRProcessor,
                                      VisionEncoderDecoderModel)
        except ImportError:
            raise SystemExit("transformers not installed.  pip install transformers")

        try:
            try:
                self.proc = TrOCRProcessor.from_pretrained(repo)
            except Exception:
                self.proc = AutoProcessor.from_pretrained(repo)
            self.model = VisionEncoderDecoderModel.from_pretrained(repo).to(device).eval()
            self.kind = "vision-encoder-decoder"
            return
        except Exception as e:
            first = e

        if not trust:
            raise SystemExit(
                f"{repo} is not a VisionEncoderDecoder model ({type(first).__name__}).\n"
                f"  It may need custom code: re-run with --hf-trust to allow\n"
                f"  trust_remote_code, AFTER reading the repo's .py files.")

        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True).to(device).eval()
        self.kind = "remote-code"

    def _one_remote(self, im):
        for meth in ("predict", "recognize", "ocr", "generate_text"):
            fn = getattr(self.model, meth, None)
            if callable(fn):
                try:
                    return str(fn(im))
                except Exception:
                    continue
        return ""

    def run(self, crops):
        from bench_ocr import clean
        torch = self.torch
        out = []
        if self.kind == "remote-code":
            for im in crops:
                try:
                    out.append(clean(self._one_remote(im.convert("RGB"))))
                except Exception:
                    out.append("")
            return out

        for i in range(0, len(crops), self.batch):
            chunk = [c.convert("RGB") for c in crops[i:i + self.batch]]
            try:
                px = self.proc(images=chunk, return_tensors="pt").pixel_values.to(self.dev)
                with torch.no_grad():
                    ids = self.model.generate(px, max_new_tokens=128)
                txt = self.proc.batch_decode(ids, skip_special_tokens=True)
                out.extend(clean(t) for t in txt)
            except Exception:
                out.extend([""] * len(chunk))
        return out


# ----------------------------------------------------------------------------

def build(spec: str, args):
    """Construct one extra engine from its command-line name, or return None."""
    if spec == "tesseract-best":
        if not args.tessdata_best:
            raise SystemExit("--tessdata-best DIR is required for the "
                             "tesseract-best engine")
        return TesseractBestEngine(args.tessdata_best, args.tess_lang)
    if spec == "surya":
        return SuryaEngine()
    if spec.startswith("hf:"):
        return HFEngine(spec[3:], device=args.hf_device, trust=args.hf_trust)
    return None
