# SHALLOW Benchmark — Windows Adaptation

![SHALLOW](assets/shallow.png)

> This is an adaptation of the [SHALLOW benchmark](https://github.com/SALT-Research/SHALLOW) by Alkis Koudounas, Moreno La Quatra, Manuel Giollo, Marco Sabato Siniscalchi, and Elena Baralis, modified for local Windows execution on the DISCOURSE study of the VOICI Speech Bank at the Centre of Excellence in Youth Mental Health, Douglas Research Centre.

This adaptation was used to evaluate three ASR models against manual transcriptions from the DISCOURSE study:

<div align="center">

| Model | HuggingFace |
|---|---|
| Whisper Large v2 | [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) |
| Canary 1B v2 | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) |
| Parakeet TDT 1.1B | [nvidia/parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) |

</div>

Each model's automatic transcriptions are compared against a manual ground truth transcription using the SHALLOW hallucination metrics (LF, PF, MH, SH, WER).

---

## Changes from original

All metrics (LF, PF, MH, SH, WER) and their formulas are unchanged. The following are infrastructure and compatibility changes only.

Metrics are computed on the **intersection** of ground truth and prediction segment IDs — segments present in only one file are excluded.

<div align="center">

| File | What changed | Why |
|---|---|---|
| `morphological.py` | `LanguageToolPublicAPI` → `LanguageTool` | The original sent text to LanguageTool's remote public server. Replaced with a local Java process so the benchmark runs offline with no external dependency. |
| `morphological.py` | Dependency tree comparison → constituency parse + Jaccard | The original compared full spaCy dependency parses (head/dep/token tuples per word). Replaced with benepar constituency parses, extracting non-terminal phrase labels (NP, VP, etc.) and computing Jaccard distance between their sets. |
| `morphological.py` | Error categorisation via `match.message` → `match.category` / `match.ruleId` | The original searched for keywords in the human-readable error message string. Replaced with the structured `category` and `ruleId` fields that LanguageTool provides for programmatic use. |
| `morphological.py` | LanguageTool and spaCy+benepar initialised once at startup | The original created a new LanguageTool Java process and loaded the spaCy model on every example. Both are now loaded once when the benchmark starts and reused throughout. |
| `semantic.py` | BERTScore loaded once at init | The original called `evaluate.load("bertscore")` inside the scoring function, reloading it from disk on every example. Moved to init so it is loaded once. |
| `semantic.py` | NLI pipeline device resolved dynamically | The original hardcoded `device=0`, assuming a CUDA GPU is always available. Now resolves the device based on what is actually available (CUDA or CPU). |
| `semantic.py` | Added in-memory embedding cache (`_emb_cache`) | The original recomputed BERT embeddings for every word window, including repeated texts, resulting in O(N²) forward passes per example. The cache stores each embedding the first time it is computed and reuses it for subsequent occurrences. |
| `fabrications.py` | `compute_measures` → `jiwer.process_words` shim | `compute_measures` was removed in jiwer 3.0. Replaced with a compatibility shim using `process_words` that returns the same values. |
| `main.py` | `multiprocessing.Pool` → sequential per-example loop | The pool caused deadlocks on Windows due to interactions between LanguageTool's Java subprocess, HuggingFace tokenizer threads, and Python's `spawn` start method. |
| `requirements.txt` | Created from scratch with pinned versions | The original repository did not ship a pinned requirements file. numpy 2.x is incompatible with pandas 1.x, and certain transformers versions caused issues with the models used. Versions are pinned to a confirmed working combination. |

</div>

## Additional scripts

- `run_shallow.py` — orchestrates the full benchmark pipeline locally on Windows, replacing the original `compute_shallow_scores.sh`
- `compute_mh.py` — standalone script to compute real MH scores using local LanguageTool + benepar on existing CSVs
- `patch_mh.py` — patches MH scores into the combined and summary CSVs after `compute_mh.py` finishes

---

## Installation

### Prerequisites

- Python 3.10
- Windows
- Java >= 17 (required for LanguageTool MH scoring)

```bash
conda create -n shallow python=3.10 -y
conda activate shallow
conda install -c conda-forge openjdk=17 -y
```

### Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import benepar; benepar.download('benepar_en3')"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
           nltk.download('averaged_perceptron_tagger'); \
           nltk.download('averaged_perceptron_tagger_eng')"
```

### Pre-warm LanguageTool (one-time, downloads ~255 MB)

```bash
python -c "import language_tool_python; t = language_tool_python.LanguageTool('en-US'); t.close(); print('OK')"
```

---

## Folder Structure

```
SHALLOW/
├── src/
│   ├── main.py
│   ├── shallow.py
│   ├── fabrications.py
│   ├── morphological.py
│   ├── semantic.py
│   └── utils.py
├── data/
│   └── shallow_format/
│       ├── manual_shallow.txt
│       ├── whisper_shallow.txt
│       ├── canary_shallow.txt
│       └── parakeet_shallow.txt
├── results/                 ← auto-created on first run
├── run_shallow.py
├── compute_mh.py
├── patch_mh.py
├── requirements.txt
└── README.md
```

---

## Usage

### Step 1 — Run the benchmark (LF, PF, SH, WER)

```bash
python run_shallow.py \
    --shallow_format_dir ./data/shallow_format \
    --output_dir ./results
```

Optional arguments:
```
--models    whisper canary parakeet    (default: all three)
```

### Step 2 — Compute real MH scores

```bash
python compute_mh.py --results_dir ./results
```

### Step 3 — Patch MH into combined and summary CSVs

```bash
python patch_mh.py --results_dir ./results
```

### Preparing your data

You need one plain text file per model plus one for the manual ground truth. Each file must have one segment per line in the format `<segment_id>: <transcription>`. Segment IDs must match exactly across all files — only segments present in both the ground truth file and the model file will be evaluated; unmatched segments are silently excluded.

```
SEGMENT_001: this is the first transcription
SEGMENT_002: this is the second transcription
```

The following file names are examples based on the models used in this study. You can use any model name — just name your file `<modelname>_shallow.txt` and pass `--models <modelname>` to `run_shallow.py`.

```
manual_shallow.txt       ← ground truth (required, always this name)
whisper_shallow.txt      ← example: Whisper hypotheses
canary_shallow.txt       ← example: Canary hypotheses
parakeet_shallow.txt     ← example: Parakeet hypotheses
```

Place all files in `data/shallow_format/` before running Step 1. The data files are not included in this repository.

---

## Authors

- **Nadine El-Mufti** · [nadine.el-mufti@mail.mcgill.ca](mailto:nadine.el-mufti@mail.mcgill.ca)
- **Dr. Alban Voppel** · [alban.voppel@mail.mcgill.ca](mailto:alban.voppel@mail.mcgill.ca)
- **Dr. Lena Palaniyappan** · [lena.palaniyappan@mcgill.ca](mailto:lena.palaniyappan@mcgill.ca)

Centre of Excellence in Youth Mental Health, Douglas Research Centre — McGill University

*For questions regarding this adaptation, contact Nadine El-Mufti.*

---

## Attribution & Citation

The SHALLOW benchmark methodology, metrics, and original codebase are the work of Alkis Koudounas, Moreno La Quatra, Manuel Giollo, Sabato Marco Siniscalchi, and Elena Baralis. This repository contains only an adaptation of [their implementation](https://github.com/SALT-Research/SHALLOW) for local Windows execution. All intellectual ownership of the benchmark itself remains with the original authors.

If you use this adaptation, please cite the original SHALLOW paper for the benchmark methodology, and reference this repository for the adapted implementation:

```bibtex
@article{koudounas2025shallow,
  title     = {Hallucination Benchmark for Speech Foundation Models},
  author    = {Koudounas, Alkis and La Quatra, Moreno and Giollo, Manuel and Siniscalchi, Sabato Marco and Baralis, Elena},
  journal   = {arXiv preprint arXiv:2510.16567},
  year      = {2025}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
