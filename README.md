# Khmer OCR benchmark harness

Evaluation code and per-line outputs for
**"Toward a Reproducible, Low-Cost Baseline for Khmer Optical Character Recognition."**

This repository lets you check the paper's results, and re-run the comparison on your own
engine. The trained models live separately, on the Hugging Face Hub:
**[vattanac/khmer-ocr-crnn-ctc](https://huggingface.co/vattanac/khmer-ocr-crnn-ctc)**.

## Verify the paper in one command

No models, no images, no dataset needed. This rescores the released per-line outputs:

```bash
git clone https://github.com/vattanac/khmer-ocr-benchmark-harness
cd khmer-ocr-benchmark-harness
python verify_tables.py
```

Expected output:

```
Table I  Document regions (KHOB level 1; 1,862 regions)
  engine              CER    paper    exact    paper
  ours-lm          0.1692   0.1692    0.287    0.287   ok
  ours             0.1978   0.1978    0.262    0.262   ok
  surya            0.3167   0.3167    0.203    0.203   ok
  tess-best        0.3941   0.3941    0.149    0.149   ok
  tesseract        0.4318   0.4318    0.145    0.145   ok
...
All values reproduce the paper within 4 decimal places.
```

If you disagree with the scoring rather than the outputs, `khmer_metrics.py` is about
forty lines and is the only thing that turns predictions into the numbers in the paper.

## What is here

| File | Purpose |
|---|---|
| `verify_tables.py` | Recompute Tables I and II from the released predictions |
| `khmer_metrics.py` | Cluster-aware (KHCWER-style) grapheme CER and exact-line accuracy |
| `khmer_ocr.py` | Model architecture and decoders, so the harness can run our model |
| `bench_ocr.py` | The benchmark runner: crops, engines, scoring, per-task breakdown |
| `engines_extra.py` | Wrappers for Tesseract `best`, Surya, and Hugging Face models |
| `analyze_degeneration.py` | Table III: repetition loops, over-length output, script drift |
| `reorder_lm.py` | The language-model ordering repair, and its evaluation |
| `eval_handwriting.py` | Table IV: handwriting evaluation |
| `predictions/level1/`, `predictions/level2/` | Per-line `prediction<TAB>reference` for all five configurations |

There is **no training code, no data-rendering pipeline and no corpus code** here. Those
are not part of the evaluation and are not released.

## Reproducing from images

To regenerate the predictions rather than rescore them you need three things the licences
do not let us redistribute:

1. **KHOB**, the benchmark data, from
   [EKYCSolutions/khmer-ocr-benchmark-dataset](https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset) (MIT).
2. **The models**, from
   [vattanac/khmer-ocr-crnn-ctc](https://huggingface.co/vattanac/khmer-ocr-crnn-ctc) (Apache 2.0).
3. **The competing engines**: `pip install pytesseract surya-ocr`, plus
   `brew install tesseract tesseract-lang llama.cpp`, and the full-precision Khmer model:

```bash
mkdir -p ~/tessdata_best && curl -L -o ~/tessdata_best/khm.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/khm.traineddata
```

Then:

```bash
python bench_ocr.py \
  --data path/to/khob/level-1 \
  --ckpt khmer_ocr_document.pt --lm khmer_lm.pkl.gz \
  --alpha 0.35 --beta 1.5 --lm-reorder \
  --engines ours,ours-lm,tesseract,tesseract-best,surya \
  --tessdata-best ~/tessdata_best --out bench_out_l1
```

Handwriting (Table IV):

```bash
python eval_handwriting.py --split testset.parquet \
  --ckpt khmer_ocr_handwriting.pt --lm khmer_lm.pkl.gz
```

## Two fairness corrections worth knowing about

Both are in `engines_extra.py`, and both change the results substantially. If you compare
Khmer OCR engines, you probably want them.

**Tesseract.** The `khm` pack that package managers install is the integer-quantised
*fast* model. The full-precision `tessdata_best` model is a different, more accurate
build: 43.2% to 39.4% CER on level 1. We report both. Everything runs with `--psm 7`
(single text line).

**Surya.** Its layout stage returned no text region for roughly 68% of short single-line
strips in our runs, so in default mode most crops are simply never recognised. We pass a
synthetic single-line layout declaring the whole crop as one text region, which is the
assumption every other engine already gets. On a 20-region probe this moved CER from 1.33
to 0.18 and time per line from 72 s to 0.7 s. Benchmarking Surya without this is
measuring its layout stage, not its recogniser.

## Scoring

Khmer writes a syllable as a base consonant plus optional COENG subscripts and vowel
signs, rendered as one unit. Scoring per Unicode code point therefore flatters a
recogniser, because one misplaced diacritic can count as several errors. We align over
orthographic clusters instead, following [KHCWER](https://github.com/keosaly/KHCWER):

```
CER = (S + D + I) / N
```

with S, D, I substituted, deleted and inserted clusters and N the clusters in the
reference. `khmer_metrics.py` also reports code-point CER, for comparison with work that
scores that way. **The two are not interchangeable**, and a code-point number will look
better than it is.

This matters when comparing against published figures: reference [3] in the paper reports
lower KHOB error for Tesseract (9.19%) and Surya (17.69%) than we measure. We have not
established the cause; the error unit and the evaluation unit (page versus line crop) are
the obvious candidates. The per-line outputs are released here so the difference can be
examined rather than argued about.

## Requirements

```bash
pip install -r requirements.txt
```

`verify_tables.py` needs only Python 3.9+. Running the benchmark additionally needs
`torch`, `pillow`, `numpy`, and whichever engines you are comparing.

## Licence and credits

Apache 2.0 (see `LICENSE`).

- **KHOB benchmark**: EKYC Solutions, with Prudential Life Assurance PLC and Paragon
  International University. MIT licence.
- **Khmer handwriting dataset (KH)**: Soy Vitou, Y Kimly, Lany Malis and Chean Botum
  (advisor Kor Sokchea), Faculty of Engineering, Royal University of Phnom Penh. MIT licence.
- **KHCWER** metric, **Tesseract**, and **Surya**. Tesseract and Surya are capable
  general-purpose systems; the comparison here is Khmer-specific and line-level.

## Citation

```bibtex
@misc{sim2026khmerocr,
  title  = {Toward a Reproducible, Low-Cost Baseline for Khmer Optical Character Recognition},
  author = {Sim, Vattanac},
  year   = {2026},
  note   = {Preprint}
}
```

Corrections are welcome, particularly reproductions that disagree with the numbers above.
