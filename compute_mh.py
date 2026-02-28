#!/usr/bin/env python3

# Script: compute_mh.py
# Author: Nadine El-Mufti (2026)

"""
Compute real Morphological Hallucination (MH) scores on existing SHALLOW CSVs.

Self-contained: all MH logic is inlined — no morphological.py import needed.
Cross-platform: Java PATH is auto-detected, no hardcoded user paths.

Run this after run_shallow.py finishes.

Folder structure (run from the SHALLOW repo root):
    SHALLOW/                      ← run scripts from here
    ├── results/
    │   ├── shallow_whisper/
    │   │   ├── shallow_metrics_discourse_whisper.csv
    │   │   └── shallow_stats_discourse_whisper.json
    │   ├── shallow_canary/   (same)
    │   └── shallow_parakeet/ (same)
    └── compute_mh.py         <- this script

Run:
    python compute_mh.py
    python compute_mh.py --results_dir ./results --models whisper canary parakeet
"""

import os
import sys
import json
import time
import platform
import re
import shutil
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Java PATH auto-detection (Windows only)
# -----------------------------------------------------------------------------

def _fix_java_path_windows():
    """
    On Windows, java.exe is often not on PATH unless the conda env bin dir is
    added manually. We try common locations and add the first valid one.
    Works for any user or conda installation — no hardcoded paths.
    """
    if platform.system() != 'Windows':
        return
    if shutil.which('java'):
        return

    candidates = [
        # conda env bin dir — sibling to the running python.exe
        os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'),
        os.path.join(os.path.dirname(sys.executable), '..', 'Library', 'bin'),
        # common standalone Java installs
        r'C:\Program Files\Java\jdk-17\bin',
        r'C:\Program Files\Java\jdk-21\bin',
        r'C:\Program Files\Eclipse Adoptium\jdk-17\bin',
    ]

    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(os.path.join(candidate, 'java.exe')):
            os.environ['PATH'] = candidate + os.pathsep + os.environ.get('PATH', '')
            print(f'  Java found and added to PATH: {candidate}')
            return

    print('  Warning: java.exe not found — LanguageTool may fail.')
    print('  Fix: conda install -c conda-forge openjdk=17 -y')


_fix_java_path_windows()


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Compute real MH scores on SHALLOW CSVs')
    parser.add_argument(
        '--results_dir',
        default='./results',
        help='Path to results folder (default: ./results)'
        )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['whisper', 'canary', 'parakeet'],
        help='Models to process (default: whisper canary parakeet)'
        )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Structural divergence via benepar parse trees
# -----------------------------------------------------------------------------

def _get_structural_divergence(ref, hyp, nlp):
    """
    Measure syntactic divergence between ref and hyp using constituency parse
    trees from benepar. Returns a float in [0, 1]: 0 = identical structure,
    1 = completely different.

    Method: Jaccard distance on the sets of non-terminal labels (NP, VP, etc.)
    extracted from the top-level parse string of the first sentence.
    """
    try:
        ref_doc = nlp(ref[:512])
        hyp_doc = nlp(hyp[:512])
        ref_sents = list(ref_doc.sents)
        hyp_sents = list(hyp_doc.sents)
        if not ref_sents or not hyp_sents:
            return 0.0
        ref_tree = ref_sents[0]._.parse_string if ref_sents[0]._.parse_string else ''
        hyp_tree  = hyp_sents[0]._.parse_string if hyp_sents[0]._.parse_string else ''
        if not ref_tree or not hyp_tree:
            return 0.0
        ref_labels = set(re.findall(r'\(([A-Z]+)', ref_tree))
        hyp_labels  = set(re.findall(r'\(([A-Z]+)', hyp_tree))
        if not ref_labels and not hyp_labels:
            return 0.0
        union   = ref_labels | hyp_labels
        jaccard = len(ref_labels & hyp_labels) / len(union)
        return round(1.0 - jaccard, 4)
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# Grammar errors via LanguageTool
# -----------------------------------------------------------------------------

def _get_grammar_errors(text, tool):
    """
    Run LanguageTool on the hypothesis text and categorize matches into
    spelling, grammar, and punctuation error counts.
    """
    try:
        matches = tool.check(text)
    except Exception:
        return {
            'total_errors': 0,
            'error_categories': {'spelling': 0, 'grammar': 0, 'punctuation': 0}
            }

    spelling = grammar = punctuation = 0
    for match in matches:
        cat     = (match.category or '').lower()
        rule_id = (match.ruleId   or '').lower()
        if 'spell' in cat or 'typo' in cat or 'misspell' in rule_id:
            spelling += 1
        elif 'punct' in cat or 'comma' in cat or 'period' in cat:
            punctuation += 1
        else:
            grammar += 1

    return {
        'total_errors': spelling + grammar + punctuation,
        'error_categories': {
            'spelling':    spelling,
            'grammar':     grammar,
            'punctuation': punctuation
            }
        }


# -----------------------------------------------------------------------------
# Tool initialization
# -----------------------------------------------------------------------------

def init_tools():
    """
    Initialize spacy + benepar (structural divergence) and LanguageTool (grammar
    errors). Each is initialized once and reused across all examples and models.
    Returns (nlp, tool) — either may be None if initialization fails.
    """
    import language_tool_python
    import spacy

    print('  Loading spacy + benepar...')
    try:
        nlp = spacy.load('en_core_web_sm')
        if 'benepar' not in nlp.pipe_names:
            nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        # Smoke test
        doc = nlp('The cat sat on the mat.')
        _ = list(doc.sents)[0]._.parse_string
        print('  spacy + benepar ready')
    except Exception as e:
        print(f'  Warning: spacy/benepar failed: {e}')
        print('  Structural divergence will be 0 for all examples.')
        nlp = None

    print('  Starting LanguageTool (Java)...')
    try:
        tool = language_tool_python.LanguageTool('en-US')
        test = tool.check('This are wrong.')
        print(f'  LanguageTool ready (smoke test: {len(test)} errors found)')
    except Exception as e:
        print(f'  Warning: LanguageTool failed: {e}')
        print('  Grammar errors will be 0 for all examples.')
        tool = None

    return nlp, tool


# -----------------------------------------------------------------------------
# Per-example MH scoring
# -----------------------------------------------------------------------------

def compute_mh_score(ref, hyp, nlp, tool):
    sd = _get_structural_divergence(ref, hyp, nlp) if nlp and ref.strip() and hyp.strip() else 0.0
    ge = _get_grammar_errors(hyp, tool) if tool and hyp.strip() else \
         {'total_errors': 0, 'error_categories': {'spelling': 0, 'grammar': 0, 'punctuation': 0}}
    return sd, ge


def gramm_score_row(row):
    """
    Weighted grammar error score normalized by hypothesis length.
    Weights: grammar=0.4, spelling=0.4, punctuation=0.2.
    """
    n = len(str(row['hyp']).split())
    if n == 0:
        return 0.0
    return (
        0.4 * row['gramm_errors_grammar'] +
        0.4 * row['gramm_errors_spelling'] +
        0.2 * row['gramm_errors_punctuation']
        ) / n


# -----------------------------------------------------------------------------
# Per-model processing
# -----------------------------------------------------------------------------

def process_model(model, nlp, tool, results_dir):
    """
    Load existing CSV for a model, compute real MH scores row by row,
    patch MH columns, and save updated CSV + JSON.
    """
    model_dir = os.path.join(results_dir, f'shallow_{model}')
    csv_path  = os.path.join(model_dir, f'shallow_metrics_discourse_{model}.csv')
    json_path = os.path.join(model_dir, f'shallow_stats_discourse_{model}.json')

    if not os.path.exists(csv_path):
        print(f'  Warning: CSV not found: {csv_path} — skipping {model}')
        return

    print(f'  Loading {csv_path} ...')
    df = pd.read_csv(csv_path)
    print(f'  {len(df)} rows loaded')

    sds, totals, spellings, grammars, puncts = [], [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'MH {model}', file=sys.stdout):
        ref = str(row.get('ref', '')).strip()
        hyp = str(row.get('hyp', '')).strip()
        try:
            sd, ge = compute_mh_score(ref, hyp, nlp, tool)
        except Exception as e:
            print(f'\n  Error on row {_}: {e}')
            sd = 0.0
            ge = {'total_errors': 0, 'error_categories': {'spelling': 0, 'grammar': 0, 'punctuation': 0}}

        sds.append(sd)
        totals.append(ge['total_errors'])
        spellings.append(ge['error_categories']['spelling'])
        grammars.append(ge['error_categories']['grammar'])
        puncts.append(ge['error_categories']['punctuation'])

    # Patch MH columns
    df['structural_divergence']             = sds
    df['gramm_errors']                      = totals
    df['gramm_errors_spelling']             = spellings
    df['gramm_errors_grammar']              = grammars
    df['gramm_errors_punctuation']          = puncts
    df['grammatical_errors_score']          = df.apply(gramm_score_row, axis=1)
    df['structural_divergence_score']       = df['structural_divergence']
    df['morphological_hallucination_score'] = (
        0.4 * df['structural_divergence_score'] +
        0.6 * df['grammatical_errors_score']
        )

    df.to_csv(csv_path, index=False)
    print(f'  CSV saved: {csv_path}')

    mh_mean = round(100 * float(np.mean(df['morphological_hallucination_score'])), 2)
    sd_mean = round(float(np.mean(df['structural_divergence'])), 4)
    ge_mean = round(float(np.mean(df['gramm_errors'])), 4)
    print(f'  MH={mh_mean}%  SD={sd_mean}  avg_errors={ge_mean}')

    # Update JSON stats
    stats = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            stats = json.load(f)
    stats['morphological_hallucination_score'] = mh_mean
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f'  JSON saved: {json_path}')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)

    print('=' * 80)
    print('COMPUTE MH SCORES — LanguageTool (local Java) + benepar')
    print('=' * 80)
    print(f'  Platform:    {platform.system()} {platform.machine()}')
    print(f'  Python:      {sys.version.split()[0]}')
    print(f'  Results dir: {results_dir}')
    print(f'  Models:      {args.models}')

    print('\n' + '-' * 80)
    print('Initializing tools')
    print('-' * 80)

    t0 = time.time()
    nlp, tool = init_tools()
    print(f'\nTools ready in {time.time() - t0:.1f}s')

    for model in args.models:
        print('\n' + '-' * 80)
        print(f'Model: {model}')
        print('-' * 80)
        t_model = time.time()
        process_model(model, nlp, tool, results_dir)
        print(f'  Done in {(time.time() - t_model) / 60:.1f} min')

    if tool is not None:
        print('\nShutting down LanguageTool...')
        try:
            tool.close()
        except Exception:
            pass

    print('\n' + '=' * 80)
    print('COMPUTE MH SCORES COMPLETE')
    print('=' * 80)

    for model in args.models:
        json_path = os.path.join(
            results_dir, f'shallow_{model}', f'shallow_stats_discourse_{model}.json'
            )
        if os.path.exists(json_path):
            with open(json_path) as f:
                stats = json.load(f)
            print(f'\n  {model}:')
            for k, v in stats.items():
                print(f'    {k}: {v}')


if __name__ == '__main__':
    main()