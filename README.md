# Signal-Check: Training-Free Watermark Defense against MLLM-Guided Semantic Editing

Reference implementation for the paper:

> **Image Watermarking Robustness under MLLM-Guided Semantic Editing: Benchmark, Defense, and Diagnosis**

This repository contains the code for (i) generating a semantic-editing attack benchmark driven
by a multimodal LLM, (ii) the training-free **signal-check** defense, and (iii) the analysis and
figure-generation scripts.

---

## What the defense does

A classical DCT-QIM watermark embeds one payload bit per 8x8 block. We additionally embed a
pseudo-random **check bit** into a second mid-frequency coefficient of the same block:

| | payload | check |
|---|---|---|
| DCT coefficient | `(4,1)` | `(3,2)` |
| Purpose | carries the message | detects damage |

At extraction, blocks whose check bit is inconsistent are assumed damaged and their payload vote
is down-weighted:

- **Soft confidence** -- the weight is the demodulation confidence of the check bit,
  `w_b = 1 - d_best / (d_best + d_other + eps)`, so partially damaged blocks are attenuated
  rather than discarded.
- **Spatial dilation** -- semantic damage is spatially contiguous, so blocks adjacent to a
  failing block are also down-weighted (x0.3).

The result is a defense that is **training-free**, needs no labelled data, and is model-agnostic:
it adds one redundant bit and a confidence weight to any block-wise embedder.

---

## Installation

```bash
pip install torch diffusers transformers opencv-python numpy pillow matplotlib scipy ultralytics openai
```

A single consumer GPU (tested on an RTX 4060 Ti 16 GB) is sufficient.

## Configuration

Copy the template and fill in your API keys:

```bash
cp .mllm_env.example .mllm_env
```

`.mllm_env` is git-ignored and must never be committed.

## Data

The attack benchmark is built from **COCO val2017** and **DIV2K**. Point the loader at your local
copies (see `code/run_idea_probe.py`, `DATA_SOURCES`).

---

## Pipeline

```
code/
|- dct_watermark.py                  # DCT-QIM embed / extract primitives
|- run_mllm_benchmark.py             # MLLM attack generation + diffusion execution
|- run_signal_ra.py                  # signal-check defense (uniform / v1 / v2 / oracle)
|- run_second_mllm_attacker.py       # attack with a *second* MLLM (cross-model check)
|- analyze_second_mllm.py            # attack-severity stratification
|
|- run_ecc_vs_checkbit.py            # check bit vs Hamming(7,4) error-correcting code
|- run_jpeg_cascade.py               # robustness to post-edit JPEG re-encoding
|- run_full_inpaint_variant.py       # unmasked (global re-rendering) attack variant
|- run_capacity_curve.py             # 32/64/128/256-bit payload sweep
|- run_ablation_coef.py              # design-space ablations
|- run_diagnosis_quant.py            # quantified reliability of MLLM diagnosis
|
|- run_sota_comparison.py            # comparison against trained baselines
|- make_figures.py                   # main figures
|- make_graphical_abstract.py        # graphical abstract
`- make_supplementary.py             # supplementary figures
```

### Reproducing the defense evaluation

```bash
cd code
python run_signal_ra.py --n 500 --msglen 256 --delta 15
```

### Cross-attacker check

```bash
cd code
python run_second_mllm_attacker.py --provider qwen --n 100
python analyze_second_mllm.py
```

`--provider` selects the attacker MLLM; the script reads `<PROVIDER>_API_KEY`,
`<PROVIDER>_BASE_URL` and `<PROVIDER>_MODEL` from `.mllm_env`.

---

## Key findings

1. **Semantic edits damage the watermark outside the edited region.** A bounding-box oracle that
   discards only the declared edit box underperforms the check-bit defense, because the damage
   leaks into neighbouring blocks.

2. **The check bit is not merely redundancy.** A Hamming(7,4) code carrying *more* redundancy
   performs *worse* than the undefended baseline, because error-correcting codes assume
   independent errors while semantic damage is spatially clustered.

3. **The gain is attacker-agnostic once severity is controlled.** Stratifying by the fraction of
   failing blocks, the high-damage stratum shows gains of +10.8pt (mimo-v2.5) and +10.5pt
   (Qwen3-VL-Plus) -- a difference of 0.3pt.

4. **There is an explicit operating envelope.** The defense works when an edit leaves a
   recoverable fraction of blocks intact; once >~35% of blocks are damaged, no block-voting
   scheme (trained or not) can reconstruct a 256-bit message. Under *global* re-rendering
   (unmasked inpainting) all training-free schemes fall to chance.

---

## License

Code released for research reproducibility. Third-party models (Stable Diffusion inpainting,
InstructPix2Pix) and datasets (COCO, DIV2K) retain their original licenses.
