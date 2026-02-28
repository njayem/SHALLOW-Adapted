#!/usr/bin/env python3

# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

# Script: fabrications.py
# Modified by: Nadine El-Mufti (2026)

import jiwer as _jiwer
import numpy as np
import jellyfish
import torch
import spacy


# -----------------------------------------------------------------------------
# jiwer compatibility shim
# -----------------------------------------------------------------------------

def compute_measures(truth, hypothesis):
    """
    Compatibility shim for jiwer >= 3.0.
    jiwer 3.x removed compute_measures() in favour of process_words().
    This wrapper preserves the original return format used by SHALLOW.
    """
    out = _jiwer.process_words(truth, hypothesis)
    return {
        'hits':          out.hits,
        'substitutions': out.substitutions,
        'deletions':     out.deletions,
        'insertions':    out.insertions,
        'wer':           out.wer,
        'mer':           out.mer,
        'wil':           out.wil,
        'wip':           out.wip,
        }


# -----------------------------------------------------------------------------
# FabricationAnalyzer
# -----------------------------------------------------------------------------

class FabricationAnalyzer:
    """
    Analyzes lexical and phonetic fabrications between reference and hypothesis
    transcriptions.
    Lexical: insertion / deletion / substitution ratios via jiwer.
    Phonetic: Hamming, Levenshtein, and Jaro-Winkler distances on metaphone
              representations via jellyfish.
    """

    def __init__(self, device='cuda'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------------
    # Lexical fabrications
    # -------------------------------------------------------------------------

    def compute_lexical_fabrications(self, reference, hypothesis):
        """
        Compute insertion, deletion, and substitution ratios between reference
        and hypothesis. Returns a dict with counts and ratios for each type.
        """
        if reference == hypothesis:
            return {
                'insertions_count':    0, 'insertions_ratio':    0.0,
                'deletions_count':     0, 'deletions_ratio':     0.0,
                'substitutions_count': 0, 'substitutions_ratio': 0.0
                }

        if len(reference) == 0:
            ins_count = len(hypothesis.split())
            return {
                'insertions_count':    ins_count,
                'insertions_ratio':    1.0 if ins_count > 0 else 0.0,
                'deletions_count':     0, 'deletions_ratio':     0.0,
                'substitutions_count': 0, 'substitutions_ratio': 0.0
                }

        if len(hypothesis) == 0:
            del_count = len(reference.split())
            return {
                'insertions_count':    0, 'insertions_ratio':    0.0,
                'deletions_count':     del_count,
                'deletions_ratio':     1.0 if del_count > 0 else 0.0,
                'substitutions_count': 0, 'substitutions_ratio': 0.0
                }

        measures = compute_measures(reference, hypothesis)
        ins  = measures['insertions']
        dels = measures['deletions']
        subs = measures['substitutions']
        n_hyp = len(hypothesis.split())
        n_ref = len(reference.split())

        return {
            'insertions_count':    ins,
            'insertions_ratio':    ins  / n_hyp if n_hyp > 0 else 0,
            'deletions_count':     dels,
            'deletions_ratio':     dels / n_ref if n_ref > 0 else 0,
            'substitutions_count': subs,
            'substitutions_ratio': subs / n_ref if n_ref > 0 else 0
            }

    # -------------------------------------------------------------------------
    # Phonetic fabrications
    # -------------------------------------------------------------------------

    def compute_phonetic_fabrications(self, reference, hypothesis):
        """
        Compute phonetic distances between reference and hypothesis using
        metaphone representations. Returns Hamming (normalized), Levenshtein
        (normalized), and Jaro-Winkler similarity.
        """
        if reference == hypothesis:
            return {
                'hamming':       0.0,
                'levenshtein':   0.0,
                'jaro_winkler':  1.0
                }

        ref_meta = jellyfish.metaphone(reference)
        hyp_meta = jellyfish.metaphone(hypothesis)
        max_len  = max(len(ref_meta), len(hyp_meta), 1)

        hamm  = jellyfish.hamming_distance(ref_meta, hyp_meta)
        leven = jellyfish.levenshtein_distance(ref_meta, hyp_meta)

        return {
            'hamming':      hamm  / max_len if hamm  is not None else 0,
            'levenshtein':  leven / max_len if leven is not None else 0,
            'jaro_winkler': jellyfish.jaro_winkler_similarity(ref_meta, hyp_meta)
            }