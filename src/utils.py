#!/usr/bin/env python3

# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

# Script: utils.py
# Modified by: Nadine El-Mufti (2026)

from whisper_normalizer.english import EnglishTextNormalizer
from jiwer import wer
import argparse
import re


# -----------------------------------------------------------------------------
# GigaSpeech constants
# -----------------------------------------------------------------------------

PUNCTUATION_TAGS = {
    '<COMMA>':          ',',
    '<PERIOD>':         '.',
    '<QUESTIONMARK>':   '?',
    '<EXCLAMATIONPOINT>': '!'
    }

GARBAGE_TAGS = [
    '<SIL>',
    '<MUSIC>',
    '<NOISE>',
    '<OTHER>'
    ]

FILLERS = [
    'UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER'
    ]


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Shallow Benchmark')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ls',
        help='Dataset to test'
        )
    parser.add_argument(
        '--model_name',
        type=str,
        default='canary1b',
        help='Model to test'
        )
    parser.add_argument(
        '--gt_transcriptions_path',
        type=str,
        default='ls_gt.txt',
        help='Path to the ground truth transcriptions file'
        )
    parser.add_argument(
        '--predictions_path',
        type=str,
        default='ls_canary1b.txt',
        help='Path to the predictions file'
        )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/',
        help='Path to the output directory'
        )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker processes to use'
        )
    parser.add_argument(
        '--examples_limit',
        type=int,
        default=-1,
        help='Limit the number of examples to process'
        )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
        )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Transcript cleaning
# -----------------------------------------------------------------------------

def clean_transcript_gigaspeech(text, remove_punctuation=False, remove_garbage=True, remove_fillers=True):
    """Clean GigaSpeech transcripts: replace or remove punctuation tags, garbage tags, and fillers."""
    processed_text = text

    if not remove_punctuation:
        for tag, symbol in PUNCTUATION_TAGS.items():
            processed_text = processed_text.replace(' ' + tag, symbol)
            processed_text = processed_text.replace(tag + ' ', symbol + ' ')
            processed_text = processed_text.replace(tag, symbol)
    else:
        for tag in PUNCTUATION_TAGS.keys():
            processed_text = processed_text.replace(tag, '')

    if remove_garbage:
        for tag in GARBAGE_TAGS:
            processed_text = processed_text.replace(tag, '')

    if remove_fillers:
        filler_pattern = r'\b(' + '|'.join(FILLERS) + r')\b'
        processed_text = re.sub(filler_pattern, '', processed_text, flags=re.IGNORECASE)

    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text


def clean_transcript_models(text):
    """Remove model-specific tags (anything inside <>) from a transcript."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------------------------------------------------------
# WER
# -----------------------------------------------------------------------------

def safe_wer(gt, hyp):
    """
    Compute WER with special handling for empty strings.
    Both empty -> 0.0. GT empty, hyp has content -> 1.0.
    """
    if not gt.strip():
        return 0.0 if not hyp.strip() else 1.0
    return wer(gt, hyp)


# -----------------------------------------------------------------------------
# Loading transcriptions
# -----------------------------------------------------------------------------

def load_transcriptions(transcription_path):
    """
    Load transcriptions from a file in the format <segment_id: transcription>.
    Returns a dict mapping segment IDs to transcription strings.
    """
    pred_transcriptions = {}
    with open(transcription_path, 'r') as f:
        for line in f:
            line = line.strip().replace('  ', ' ')
            if line == '':
                continue
            try:
                segment, transcription = line.split(': ', 1)
                if transcription[0] in ("'", '"'):
                    transcription = transcription[1:]
                if transcription[-1] in ("'", '"'):
                    transcription = transcription[:-1]
                pred_transcriptions[segment] = transcription
            except Exception:
                segment, transcription = line.split(':')
                pred_transcriptions[segment] = ''
    return pred_transcriptions


def keep_intersection_and_sort(gt_transcriptions, pred_transcriptions):
    """Keep only keys present in both dicts and sort by key."""
    keys = set(gt_transcriptions.keys()).intersection(set(pred_transcriptions.keys()))
    gt_transcriptions   = dict(sorted({k: gt_transcriptions[k]   for k in keys}.items()))
    pred_transcriptions = dict(sorted({k: pred_transcriptions[k] for k in keys}.items()))
    return gt_transcriptions, pred_transcriptions


def load_gt_pred_transcriptions(gt_path, predictions_path):
    """
    Load, clean, and normalize ground truth and predicted transcriptions.
    Returns two lists (gt, pred) aligned by segment ID.
    """
    gt_transcriptions   = load_transcriptions(gt_path)
    pred_transcriptions = load_transcriptions(predictions_path)

    gt_transcriptions, pred_transcriptions = keep_intersection_and_sort(
        gt_transcriptions, pred_transcriptions
        )

    if 'gigaspeech' in gt_path:
        gt_transcriptions   = {k: clean_transcript_gigaspeech(v) for k, v in gt_transcriptions.items()}
        pred_transcriptions = {k: clean_transcript_gigaspeech(v) for k, v in pred_transcriptions.items()}

    gt_transcriptions   = {k: clean_transcript_models(str(v)) for k, v in gt_transcriptions.items()}
    pred_transcriptions = {k: clean_transcript_models(str(v)) for k, v in pred_transcriptions.items()}

    english_normalizer  = EnglishTextNormalizer()
    gt_transcriptions   = [english_normalizer(gt) for gt in gt_transcriptions.values()]
    pred_transcriptions = [english_normalizer(pt) for pt in pred_transcriptions.values()]

    return gt_transcriptions, pred_transcriptions