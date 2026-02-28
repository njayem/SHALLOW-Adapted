#!/usr/bin/env python3

# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

# Script: shallow.py
# Modified by: Nadine El-Mufti (2026)

from jiwer import wer
import numpy as np
import benepar
import torch
import nltk

from semantic import SemanticAnalyzer
from fabrications import FabricationAnalyzer
from morphological import MorphologicalAnalyzer


# -----------------------------------------------------------------------------
# ShallowBenchmark
# -----------------------------------------------------------------------------

class ShallowBenchmark:
    """
    Top-level benchmark class for analyzing hallucinations in ASR systems.
    Composes FabricationAnalyzer, MorphologicalAnalyzer, and SemanticAnalyzer
    to compute LF, PF, MH, and SH scores per example and across a dataset.
    """

    def __init__(
        self,
        grammar_tool_language='en-US',
        language_model='en_core_web_sm',
        local_model_name='bert-base-uncased',
        global_model_name='nli-roberta-base-v2',
        nli_model_name='facebook/bart-large-mnli',
        device=None
        ):
        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        # Download required NLTK resources
        nltk.download('punkt',                      quiet=True)
        nltk.download('stopwords',                  quiet=True)
        nltk.download('punkt_tab',                  quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        # Download benepar model
        benepar.download('benepar_en3')

        # Initialize analyzers
        self.fabrications  = FabricationAnalyzer(device=device)
        self.morphological = MorphologicalAnalyzer(
            grammar_tool_language=grammar_tool_language,
            language_model=language_model
            )
        self.semantic = SemanticAnalyzer(
            local_model_name=local_model_name,
            global_model_name=global_model_name,
            nli_model_name=nli_model_name,
            device=device
            )

    # -------------------------------------------------------------------------
    # Per-example scoring
    # -------------------------------------------------------------------------

    def __call__(self, ref, hyp):
        """
        Compute all hallucination scores for a single ref/hyp pair.
        Returns a dict with keys: wer_score, lexical_fabrication_score,
        phonetic_fabrication_score, morphological_hallucination_score,
        local_semantic_score, global_semantic_score.
        """
        return {
            'wer_score':                       self.safe_wer(ref, hyp),
            'lexical_fabrication_score':       self.fabrications.compute_lexical_fabrications(ref, hyp),
            'phonetic_fabrication_score':      self.fabrications.compute_phonetic_fabrications(ref, hyp),
            'morphological_hallucination_score': self.morphological.morphological_hallucination_score(ref, hyp),
            'local_semantic_score':            self.semantic.local_semantic_score(ref, hyp),
            'global_semantic_score':           self.semantic.global_semantic_score(ref, hyp),
            }

    # -------------------------------------------------------------------------
    # Dataset-level WER
    # -------------------------------------------------------------------------

    def compute_dataset_wer(self, gt_transcriptions, pred_transcriptions):
        """Compute corpus-level WER by concatenating all transcriptions."""
        corpus_gt   = ' '.join(gt_transcriptions)
        corpus_pred = ' '.join(pred_transcriptions)
        return round(100 * wer(corpus_gt, corpus_pred), 2)

    def safe_wer(self, reference, hypothesis):
        """WER with empty-string handling: both empty -> 0.0, GT empty -> 1.0."""
        if not reference.strip():
            return 0.0 if not hypothesis.strip() else 1.0
        return wer(reference, hypothesis)

    # -------------------------------------------------------------------------
    # Aggregated lexical fabrication
    # -------------------------------------------------------------------------

    def aggregated_lexical_fabrication_score(
        self,
        ins_ratios,
        del_ratios,
        sub_ratios,
        hypotheses,
        fillers=[
            'actually', 'literally', 'definitely', 'er', 'just', 'absolutely',
            'mmm', 'ah', 'well', 'seriously', 'basically', 'you know', 'I mean',
            'ahm', 'like', 'sort of', 'I guess', 'I suppose', 'I think', 'right',
            'ok', 'um', 'mm', 'probably', 'totally', 'kind of', 'uh', 'maybe',
            'no doubt', 'okay', 'uhm', 'really', 'so', 'for sure'
            ]
        ):
        """Weighted insertion/substitution/deletion score. Filler-only hypotheses score 1.0."""
        scores = []
        for ins, dele, sub, hyp in zip(ins_ratios, del_ratios, sub_ratios, hypotheses):
            if ins == 1 and all(word in fillers for word in hyp.split()):
                scores.append(1.0)
            else:
                scores.append(0.5 * ins + 0.3 * sub + 0.2 * dele)
        return scores

    # -------------------------------------------------------------------------
    # Aggregated phonetic fabrication
    # -------------------------------------------------------------------------

    def aggregated_phonetic_score(self, hammings, levenshteins, jaro_winklers):
        """Equal-weight average of Hamming, Levenshtein, and Jaro-Winkler distances."""
        return [(h + l + (1 - j)) / 3 for h, l, j in zip(hammings, levenshteins, jaro_winklers)]

    # -------------------------------------------------------------------------
    # Aggregated morphological hallucination
    # -------------------------------------------------------------------------

    def aggregated_grammatical_errors_score(
        self, spelling_errors, grammar_errors, punctuation_errors, hypotheses
        ):
        """
        Weighted grammar error score normalized by hypothesis length.
        Weights: grammar=0.4, spelling=0.4, punctuation=0.2.
        """
        scores = []
        for s, g, p, hyp in zip(spelling_errors, grammar_errors, punctuation_errors, hypotheses):
            try:
                n = len(hyp.split())
                scores.append((0.4 * g + 0.4 * s + 0.2 * p) / n if n > 0 else 0)
            except Exception:
                scores.append(0)
        return scores

    def aggregated_morphological_hallucination_score(
        self, syntax_divergences, spelling_errors, grammar_errors, punctuation_errors, hypotheses
        ):
        """
        MH = 0.4 * structural_divergence + 0.6 * grammatical_errors_score.
        Returns (structural_divergence_scores, grammatical_errors_scores, mh_scores).
        """
        ge = self.aggregated_grammatical_errors_score(
            spelling_errors, grammar_errors, punctuation_errors, hypotheses
            )
        return (
            syntax_divergences,
            ge,
            [0.4 * s + 0.6 * g for s, g in zip(syntax_divergences, ge)]
            )

    # -------------------------------------------------------------------------
    # Aggregated semantic hallucination
    # -------------------------------------------------------------------------

    def aggregated_local_semantic_score(self, c1s, c2s, c3s):
        """Weighted local semantic divergence: 0.5*(1-w1) + 0.3*(1-w2) + 0.2*(1-w3)."""
        return [0.5 * (1 - c1) + 0.3 * (1 - c2) + 0.2 * (1 - c3) for c1, c2, c3 in zip(c1s, c2s, c3s)]

    def aggregated_semantic_distance_score(self, cosines):
        return [1 - c for c in cosines]

    def aggregated_semantic_coherence_score(self, semantic_coherences):
        return [1 - sc for sc in semantic_coherences]

    def aggregated_global_semantic_score(self, cosines, semantic_coherences):
        """Average of semantic distance and coherence divergence scores."""
        sem_dist  = self.aggregated_semantic_distance_score(cosines)
        sem_coher = self.aggregated_semantic_coherence_score(semantic_coherences)
        return [(sd + sc) / 2 for sd, sc in zip(sem_dist, sem_coher)]

    def aggregated_semantic_score(self, local_semantic_scores, global_semantic_scores):
        """SH = 0.25 * local + 0.75 * global."""
        return [0.25 * ls + 0.75 * gs for ls, gs in zip(local_semantic_scores, global_semantic_scores)]

    # -------------------------------------------------------------------------
    # Dataset statistics
    # -------------------------------------------------------------------------

    def compute_dataset_stats(self, df):
        """Return mean * 100 for all numeric columns in df."""
        return {
            metric: round(100 * np.mean(df[metric]), 2)
            for metric in df.columns
            if df[metric].dtype != 'O'
            }