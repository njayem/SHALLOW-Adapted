#!/usr/bin/env python3

# Script: run_shallow.py
# Author: Nadine El-Mufti (2026)

"""
Run SHALLOW hallucination benchmark locally with full LanguageTool MH scoring.
Supports CUDA and CPU backends.
Platform: Windows.

ALL metrics computed: LF, PF, MH (real LanguageTool — no stubs), SH, WER.

Folder structure (run from the SHALLOW repo root):
    SHALLOW/                      <- git clone / fork root — run scripts from here
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
    ├── results/                  <- auto-created
    ├── run_shallow.py            <- this script
    ├── compute_mh.py
    └── patch_mh.py

Conda setup (run once):
    conda create -n shallow python=3.10 -y
    conda activate shallow
    conda install -c conda-forge openjdk=17 -y        <- must be 17+, not 11
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python -c "import benepar; benepar.download('benepar_en3')"
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
               nltk.download('averaged_perceptron_tagger'); \
               nltk.download('averaged_perceptron_tagger_eng')"

    # Pre-warm LanguageTool ONCE (downloads ~255 MB JAR, required before first run):
    python -c "import language_tool_python; t = language_tool_python.LanguageTool('en-US'); t.close(); print('OK')"

Run:
    python run_shallow.py \
        --shallow_format_dir ./data/shallow_format \
        --output_dir ./results
"""

import os
import sys
import platform
import shutil
import subprocess
import argparse
import json
import time
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Platform detection
# ─────────────────────────────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == 'Windows'




# -----------------------------------------------------------------------------
# Subprocess environment — prevents deadlocks on Windows
# -----------------------------------------------------------------------------

def get_subprocess_env():
    """
    Return an environment dict that prevents the most common deadlock causes
    on Windows.

    TOKENIZERS_PARALLELISM=false
        HuggingFace tokenizers spawn background threads that conflict with
        Python's multiprocessing after spawn(). Setting this to false stops
        the threads from being created at all.

    OMP_NUM_THREADS=1
        OpenMP threads conflict with Python's multiprocessing after spawn.
        Capping at 1 prevents the conflict without hurting single-example
        throughput (SHALLOW runs examples sequentially anyway).

    CUDA_VISIBLE_DEVICES=''
        Hides CUDA so torch falls back to CPU on machines without a CUDA
        driver, instead of crashing.

    PYTHONIOENCODING=utf-8
        Windows default console encoding is cp1252. Without this the
        subprocess stdout pipe can't encode Unicode characters (e.g. progress
        bar blocks) and raises UnicodeDecodeError.
    """
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['OMP_NUM_THREADS']        = '1'
    env['CUDA_VISIBLE_DEVICES']   = ''
    env['PYTHONIOENCODING']       = 'utf-8'
    return env


# ─────────────────────────────────────────────────────────────────────────────
# LanguageTool pre-warm — download JAR once before benchmark starts
# ─────────────────────────────────────────────────────────────────────────────

def prewarm_languagetool():
    """
    Download and cache the LanguageTool JAR (~255 MB) before running the
    benchmark.

    Why this matters:
        Without pre-warming, the JAR download happens silently inside the
        benchmark subprocess while it is processing the very first example.
        The benchmark appears completely frozen for 5-10 minutes with no
        output, which is indistinguishable from a deadlock.

    Cache location (Windows):
        %LOCALAPPDATA%\\language_tool_python\\
        or ~/.cache/language_tool_python/

    After the first run the JAR is cached and this function returns
    immediately on every subsequent call.
    """
    print("  Checking LanguageTool cache...")

    cache_candidates = [
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'language_tool_python'),
        os.path.join(os.path.expanduser('~'), '.cache', 'language_tool_python'),
    ]

    already_cached = any(
        os.path.isdir(p) and os.listdir(p)
        for p in cache_candidates
        if os.path.isdir(p)
    )

    if already_cached:
        print("  ✓ LanguageTool already cached — MH scoring ready")
        return True

    print("  LanguageTool JAR not found — downloading now (~255 MB, one-time only).")
    print("  This takes 1-3 minutes. Do NOT interrupt.")

    prewarm_code = (
        "import language_tool_python; "
        "t = language_tool_python.LanguageTool('en-US'); "
        "assert t.check('This are wrong.'), 'LanguageTool check failed'; "
        "t.close(); "
        "print('LanguageTool OK')"
    )

    try:
        result = subprocess.run(
            [sys.executable, '-c', prewarm_code],
            env=get_subprocess_env(),
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600,
        )
        if result.returncode == 0:
            print("  ✓ LanguageTool downloaded and verified — real MH scores enabled")
            return True
        else:
            print("  ⚠ LanguageTool pre-warm failed. MH scores may be 0.")
            print("    Debug: python -c \"import language_tool_python; language_tool_python.LanguageTool('en-US')\"")
            return False
    except subprocess.TimeoutExpired:
        print("  ⚠ LanguageTool download timed out (>10 min). Check your internet connection.")
        return False
    except Exception as e:
        print(f"  ⚠ LanguageTool pre-warm error: {e}")
        return False


def run_shallow_and_merge(
    shallow_repo,
    shallow_format_dir,
    output_dir,
    models=['whisper', 'canary', 'parakeet']
    ):
    """
        Run the SHALLOW benchmark on each model and merge results.
        Args:
            shallow_repo (str): Path to the cloned SHALLOW repository.
            shallow_format_dir (str): Directory containing shallow-format transcription files.
            output_dir (str): Directory to write results to.
            models (list): List of model names to evaluate.
        Returns:
            pd.DataFrame or None: Combined results DataFrame, or None if no results.
    """
    ## Resolve everything to absolute paths to avoid cwd issues
    shallow_repo = os.path.abspath(shallow_repo)
    shallow_format_dir = os.path.abspath(shallow_format_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("SHALLOW BENCHMARK — LOCAL EXECUTION")
    print("=" * 80)
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Repo:     {shallow_repo}")
    print(f"  Data:     {shallow_format_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Models:   {models}")
    print(f"  Metrics:  LF + PF + MH (LanguageTool) + SH + WER  [ALL ENABLED]")

    # =====================================================================
    # STEP 1: Verify prerequisites
    # =====================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Verifying prerequisites")
    print("-" * 80)

    if not os.path.exists(os.path.join(shallow_repo, 'src')):
        print(f'  Error: src/ not found at: {os.path.join(shallow_repo, "src")}')
        print('  Are you running from the SHALLOW repo root?')
        return None

    ## Check Java (required for local LanguageTool → MH metric)
    try:
        java_check = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if java_check.returncode != 0:
            raise FileNotFoundError
        java_ver = (java_check.stderr or java_check.stdout).strip().split('\n')[0]
        print(f"  ✓ Java: {java_ver}")

        # LanguageTool latest requires Java >= 17
        import re as _re
        m = _re.search(r'"(\d+)', java_ver)
        if m and int(m.group(1)) < 17:
            print(f"  ⚠ Java {m.group(1)} detected — LanguageTool requires Java >= 17.")
            print("    Fix: conda install -c conda-forge openjdk=17 -y")
            print("    Continuing, but MH scores will likely fail.")
    except FileNotFoundError:
        print("  ❌ Java not found! LanguageTool (MH metric) requires Java >= 17.")
        print("    Fix: conda install -c conda-forge openjdk=17 -y")
        return None

    # =====================================================================
    # STEP 2: Pre-warm LanguageTool (downloads JAR if not already cached)
    # =====================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Pre-warming LanguageTool (MH metric)")
    print("-" * 80)

    prewarm_languagetool()

    # =====================================================================
    # STEP 3: Run SHALLOW on each model
    # =====================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Running SHALLOW benchmark on all models")
    print("-" * 80)

    gt_path = os.path.join(shallow_format_dir, 'manual_shallow.txt')

    if not os.path.exists(gt_path):
        print(f"  ❌ Ground truth not found: {gt_path}")
        return None

    all_model_stats = {}
    all_model_dfs = []

    for model in models:
        pred_path = os.path.join(shallow_format_dir, f'{model}_shallow.txt')

        if not os.path.exists(pred_path):
            print(f"  ⚠ Skipping {model}: {pred_path} not found")
            continue

        print(f"\n  {'='*60}")
        print(f"  Running SHALLOW on: {model}")
        print(f"  {'='*60}")
        print(f"    GT:   {gt_path}")
        print(f"    Pred: {pred_path}")

        model_output = os.path.join(output_dir, f'shallow_{model}')
        os.makedirs(model_output, exist_ok=True)

        cmd = [
            sys.executable, '-u', 'main.py',
            '--dataset_name', 'discourse',
            '--model_name', model,
            '--gt_transcriptions_path', gt_path,
            '--predictions_path', pred_path,
            '--output_dir', model_output,
            '--examples_limit', '-1',
            '--num_workers', '1'
        ]

        print(f"    cmd: {' '.join(cmd)}")
        print(f"    cwd: {os.path.join(shallow_repo, 'src')}")
        t0 = time.time()

        subprocess_env = get_subprocess_env()

        # Use Popen with line-by-line streaming instead of capture_output=True.
        # capture_output=True buffers ALL output until the process exits —
        # on a 1000-example run that is 2+ hours of silence, indistinguishable
        # from a deadlock. Streaming lets you see the tqdm progress bar live.
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge stderr into stdout stream
                text=True,
                encoding='utf-8',
                errors='replace',           # don't crash on stray Unicode (Windows)
                cwd=os.path.join(shallow_repo, 'src'),
                env=subprocess_env,
            )

            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                print(f"    {line}")

            process.wait()
            returncode = process.returncode

        except KeyboardInterrupt:
            print(f"\n    ⚠ Interrupted — killing subprocess")
            process.kill()
            process.wait()
            raise
        except Exception as e:
            print(f"    ❌ Subprocess error: {e}")
            continue

        elapsed = time.time() - t0
        print(f"\n    ⏱  Finished in {elapsed/60:.1f} min (exit code {returncode})")

        if returncode != 0:
            print(f"    ❌ SHALLOW failed for {model}")
            print("    Last 20 lines:")
            for line in output_lines[-20:]:
                print(f"      {line}")
            continue

        ## Check output files
        stats_file = os.path.join(model_output, f'shallow_stats_discourse_{model}.json')
        csv_file = os.path.join(model_output, f'shallow_metrics_discourse_{model}.csv')

        if os.path.exists(stats_file):
            with open(stats_file, encoding='utf-8') as f:
                stats = json.load(f)
            all_model_stats[model] = stats
            print(f"    ✅ Stats: {json.dumps(stats, indent=2)}")

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['model'] = model
            all_model_dfs.append(df)
            print(f"    ✅ CSV: {len(df)} rows × {len(df.columns)} cols")

        ## List all files produced
        print(f"    📁 Files in {model_output}:")
        if os.path.exists(model_output):
            for fname in sorted(os.listdir(model_output)):
                fpath = os.path.join(model_output, fname)
                size = os.path.getsize(fpath)
                print(f"      {fname}  ({size/1024:.1f} KB)")

    # =====================================================================
    # STEP 4: Build summary table
    # =====================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Building summary table")
    print("-" * 80)

    if not all_model_dfs:
        print("  ❌ No SHALLOW results obtained")
        return None

    ## Combine all model results
    combined = pd.concat(all_model_dfs, ignore_index=True)
    combined_path = os.path.join(output_dir, 'shallow_all_models_combined.csv')
    combined.to_csv(combined_path, index=False)
    print(f"  ✓ Combined CSV: {combined_path} ({len(combined)} rows)")

    ## Compute summary statistics per model
    agg_cols = [
        'lexical_fabrication_score',
        'phonetic_fabrication_score',
        'morphological_hallucination_score',
        'semantic_hallucination_score',
        'wer',
        ]
    available_cols = [c for c in agg_cols if c in combined.columns]

    if available_cols:
        summary = combined.groupby('model')[available_cols].agg(['mean', 'std'])
        summary.columns = [f'{c}_{s}' for c, s in summary.columns]

        summary_formatted = pd.DataFrame(index=summary.index)
        for col in available_cols:
            mean_col, std_col = f'{col}_mean', f'{col}_std'
            if mean_col in summary.columns:
                summary_formatted[col] = summary.apply(
                    lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}", axis=1
                    )

        summary_path = os.path.join(output_dir, 'shallow_summary.csv')
        summary_formatted.to_csv(summary_path)
        print(f"  ✓ Summary: {summary_path}")

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(summary_formatted.to_string())

    # =====================================================================
    # STEP 5: Print final stats from JSON
    # =====================================================================
    if all_model_stats:
        print("\n" + "-" * 80)
        print("SHALLOW STATS (from JSON)")
        print("-" * 80)
        for model, stats in all_model_stats.items():
            print(f"\n  {model}:")
            for k, v in stats.items():
                print(f"    {k}: {v}")

    # =====================================================================
    # DONE
    # =====================================================================
    print("\n" + "=" * 80)
    print("✅ SHALLOW BENCHMARK COMPLETE")
    print("=" * 80)

    print(f"\n📁 Files created in {output_dir}:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath) and f.endswith('.csv'):
            size = os.path.getsize(fpath) / 1024
            print(f"   • {f} ({size:.1f} KB)")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHALLOW benchmark locally")
    parser.add_argument(
        "--shallow_repo",
        default=".",
        help="Path to SHALLOW repo root (default: ., i.e. current directory)"
        )
    parser.add_argument(
        "--shallow_format_dir",
        default="./data",
        help="Directory with manual_shallow.txt + model_shallow.txt files"
        )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Where to write results"
        )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper", "canary", "parakeet"],
        help="Model names to evaluate"
        )
    args = parser.parse_args()

    run_shallow_and_merge(
        shallow_repo=args.shallow_repo,
        shallow_format_dir=args.shallow_format_dir,
        output_dir=args.output_dir,
        models=args.models
        )