#!/usr/bin/env python3

# Script: patch_mh.py
# Author: Nadine El-Mufti (2026)

"""
Patch real MH scores into shallow_all_models_combined.csv and
shallow_summary.csv from the individual per-model CSVs.

Run this after compute_mh.py finishes.

Folder structure (run from the SHALLOW repo root):
    SHALLOW/                      ← run scripts from here
    ├── results/
    │   ├── shallow_all_models_combined.csv
    │   ├── shallow_summary.csv
    │   ├── shallow_whisper/shallow_metrics_discourse_whisper.csv
    │   ├── shallow_canary/shallow_metrics_discourse_canary.csv
    │   └── shallow_parakeet/shallow_metrics_discourse_parakeet.csv
    └── patch_mh.py    <- this script

Run:
    python patch_mh.py
    python patch_mh.py --results_dir ./results --models whisper canary parakeet
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Patch MH scores into combined and summary CSVs'
        )
    parser.add_argument(
        '--results_dir',
        default='./results',
        help='Path to results folder (default: ./results)'
        )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['whisper', 'canary', 'parakeet'],
        help='Models to patch (default: whisper canary parakeet)'
        )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# MH columns to patch from individual CSVs into the combined CSV
# -----------------------------------------------------------------------------

MH_COLS = [
    'structural_divergence',
    'gramm_errors',
    'gramm_errors_spelling',
    'gramm_errors_grammar',
    'gramm_errors_punctuation',
    'structural_divergence_score',
    'grammatical_errors_score',
    'morphological_hallucination_score',
]


# -----------------------------------------------------------------------------
# Patching logic
# -----------------------------------------------------------------------------

def patch_combined(combined, models, results_dir):
    """
    For each model, load the individual per-example CSV and overwrite
    all MH-related columns in the combined dataframe.
    """
    print('\n' + '-' * 80)
    print('Patching shallow_all_models_combined.csv')
    print('-' * 80)

    for model in models:
        csv_path = os.path.join(
            results_dir, f'shallow_{model}',
            f'shallow_metrics_discourse_{model}.csv'
            )
        if not os.path.exists(csv_path):
            print(f'  Warning: {csv_path} not found — skipping {model}')
            continue

        df   = pd.read_csv(csv_path)
        mask = combined['model'] == model

        if mask.sum() != len(df):
            print(f'  Warning: row count mismatch for {model} '
                  f'(combined={mask.sum()}, individual={len(df)}) — skipping')
            continue

        for col in MH_COLS:
            if col in df.columns:
                combined.loc[mask, col] = df[col].values

        mh_mean = combined.loc[mask, 'morphological_hallucination_score'].mean()
        print(f'  {model}: MH mean = {mh_mean:.4f}')

    return combined


def patch_summary(summary, combined, models):
    """
    Recompute mean +/- std MH from the patched combined dataframe and
    update the summary CSV.
    """
    print('\n' + '-' * 80)
    print('Patching shallow_summary.csv')
    print('-' * 80)

    for model in models:
        mask = combined['model'] == model
        mh   = combined.loc[mask, 'morphological_hallucination_score']
        val  = f'{mh.mean():.4f} +/- {mh.std():.4f}'
        summary.loc[summary['model'] == model, 'morphological_hallucination_score'] = val
        print(f'  {model}: {val}')

    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args        = parse_args()
    results_dir = os.path.abspath(args.results_dir)

    combined_path = os.path.join(results_dir, 'shallow_all_models_combined.csv')
    summary_path  = os.path.join(results_dir, 'shallow_summary.csv')

    print('=' * 80)
    print('PATCH MH SCORES INTO COMBINED + SUMMARY CSVs')
    print('=' * 80)
    print(f'  Results dir: {results_dir}')
    print(f'  Models:      {args.models}')

    # -------------------------------------------------------------------------
    # Load combined
    # -------------------------------------------------------------------------
    if not os.path.exists(combined_path):
        print(f'\nError: combined CSV not found: {combined_path}')
        sys.exit(1)

    combined = pd.read_csv(combined_path)
    print(f'\nLoaded combined: {len(combined)} rows')
    print('MH before:')
    print(combined.groupby('model')['morphological_hallucination_score'].mean().round(4).to_string())

    # -------------------------------------------------------------------------
    # Patch and save combined
    # -------------------------------------------------------------------------
    combined = patch_combined(combined, args.models, results_dir)
    combined.to_csv(combined_path, index=False)
    print(f'\nSaved: {combined_path}')

    # -------------------------------------------------------------------------
    # Patch and save summary
    # -------------------------------------------------------------------------
    if not os.path.exists(summary_path):
        print(f'\nWarning: summary CSV not found: {summary_path} — skipping')
    else:
        summary = pd.read_csv(summary_path)
        summary = patch_summary(summary, combined, args.models)
        summary.to_csv(summary_path, index=False)
        print(f'\nSaved: {summary_path}')

    # -------------------------------------------------------------------------
    # Final printout
    # -------------------------------------------------------------------------
    print('\n' + '=' * 80)
    print('PATCH MH COMPLETE')
    print('=' * 80)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()