#!/usr/bin/env python3

# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

# Script: morphological.py
# Modified by: Nadine El-Mufti (2026)

import language_tool_python
import benepar
import spacy
import re


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_structural_divergence(ref, hyp, nlp):
    """
    Compute structural divergence between ref and hyp using benepar parse trees.
    Returns a float between 0 (identical) and 1 (completely different).
    Method: Jaccard distance on non-terminal label sets from the parse string.
    """
    try:
        ref_doc = nlp(ref[:512])
        hyp_doc = nlp(hyp[:512])

        ref_sents = list(ref_doc.sents)
        hyp_sents = list(hyp_doc.sents)

        if not ref_sents or not hyp_sents:
            return 0.0

        ref_tree = ref_sents[0]._.parse_string if ref_sents[0]._.parse_string else ''
        hyp_tree = hyp_sents[0]._.parse_string if hyp_sents[0]._.parse_string else ''

        if not ref_tree or not hyp_tree:
            return 0.0

        ref_labels = set(re.findall(r'\(([A-Z]+)', ref_tree))
        hyp_labels = set(re.findall(r'\(([A-Z]+)', hyp_tree))

        if not ref_labels and not hyp_labels:
            return 0.0

        intersection = ref_labels & hyp_labels
        union        = ref_labels | hyp_labels
        jaccard      = len(intersection) / len(union) if union else 1.0
        return round(1.0 - jaccard, 4)

    except Exception:
        return 0.0


def _get_grammar_errors(text, tool):
    """
    Run LanguageTool on text and categorize errors into spelling, grammar,
    and punctuation counts.
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
        cat     = match.category.lower() if match.category else ''
        rule_id = match.ruleId.lower()   if match.ruleId   else ''
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
# MorphologicalAnalyzer
# -----------------------------------------------------------------------------

class MorphologicalAnalyzer:
    """
    Analyzes morphological hallucinations using LanguageTool + benepar.
    Both tools are initialized once and reused across all examples to avoid
    per-call Java restarts and model reloads.
    """

    def __init__(self, grammar_tool_language='en-US', language_model='en_core_web_sm'):
        # Load spacy + benepar
        try:
            self.nlp = spacy.load(language_model)
            if 'benepar' not in self.nlp.pipe_names:
                self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        except Exception as e:
            print(f'  Warning: spacy/benepar init failed: {e}')
            print('  Structural divergence will be 0 for all examples.')
            self.nlp = None

        # Initialize LanguageTool
        try:
            self.grammar_tool = language_tool_python.LanguageTool(grammar_tool_language)
            print(f'  LanguageTool initialized for {grammar_tool_language}')
        except Exception as e:
            print(f'  Warning: LanguageTool init failed: {e}')
            print('  Grammar errors will be 0 for all examples.')
            self.grammar_tool = None

    def morphological_hallucination_score(self, ref, hyp):
        """
        Compute morphological hallucination score for a ref/hyp pair.
        Returns dict with structural_divergence and grammatical_errors.
        """
        if self.nlp is not None and ref.strip() and hyp.strip():
            structural_divergence = _get_structural_divergence(ref, hyp, self.nlp)
        else:
            structural_divergence = 0.0

        if self.grammar_tool is not None and hyp.strip():
            grammatical_errors = _get_grammar_errors(hyp, self.grammar_tool)
        else:
            grammatical_errors = {
                'total_errors': 0,
                'error_categories': {'spelling': 0, 'grammar': 0, 'punctuation': 0}
                }

        return {
            'structural_divergence': structural_divergence,
            'grammatical_errors':    grammatical_errors
            }

    def __call__(self, ref, hyp):
        return self.morphological_hallucination_score(ref, hyp)

    def close(self):
        """Cleanly shut down the LanguageTool Java process."""
        if self.grammar_tool is not None:
            try:
                self.grammar_tool.close()
            except Exception:
                pass