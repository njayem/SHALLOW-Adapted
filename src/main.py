#!/usr/bin/env python3

# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

# Script: main.py
# Modified by: Nadine El-Mufti (2026)

from tqdm import tqdm
import pandas as pd
import torch
import json
import time
import os
import sys

from shallow import ShallowBenchmark
from utils import (
    parse_args,
    load_gt_pred_transcriptions
    )

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('evaluate').setLevel(logging.ERROR)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)


# -----------------------------------------------------------------------------
# Device detection
# -----------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    segment_ids, gt_transcriptions, pred_transcriptions = load_gt_pred_transcriptions(
        gt_path=args.gt_transcriptions_path,
        predictions_path=args.predictions_path
        )

    if args.examples_limit > 0:
            segment_ids = segment_ids[:args.examples_limit]
            gt_transcriptions  = gt_transcriptions[:args.examples_limit]
            pred_transcriptions = pred_transcriptions[:args.examples_limit]

    print(f'Loaded {len(gt_transcriptions)} examples', flush=True)
    print(f'Loaded {len(pred_transcriptions)} examples', flush=True)

    device_str = get_device()

    print(f'Dataset: {args.dataset_name}', flush=True)
    print(f'Model:   {args.model_name}', flush=True)
    print(f'Processing {len(gt_transcriptions)} examples sequentially on {device_str}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print('Initializing ShallowBenchmark...', flush=True)
    t0 = time.time()
    shallow_bench = ShallowBenchmark(
        grammar_tool_language='en-US',
        language_model='en_core_web_sm',
        local_model_name='bert-base-uncased',
        global_model_name='nli-roberta-base-v2',
        nli_model_name='facebook/bart-large-mnli',
        device=device_str
        )
    print(f'ShallowBenchmark initialized ({time.time() - t0:.1f}s)', flush=True)

    all_results  = []
    total_errors = 0

    # tqdm writes to stdout so the progress bar is visible through the subprocess
    # pipe on Windows (tqdm defaults to stderr, which may be buffered separately)
    for idx, (ref, hyp) in tqdm(
        enumerate(zip(gt_transcriptions, pred_transcriptions)),
        total=len(gt_transcriptions),
        desc='SHALLOW',
        file=sys.stdout,
        dynamic_ncols=True
        ):
        if str(ref).strip() == str(hyp).strip():
            all_results.append({
                'ref': ref,
                'hyp': hyp,
                'wer': 0,
                'ins_count': 0,
                'ins_ratio': 0.0,
                'del_count': 0,
                'del_ratio': 0.0,
                'sub_count': 0,
                'sub_ratio': 0.0,
                'phonetic_hamming': 0,
                'phonetic_levenshtein': 0,
                'phonetic_jaro_winkler': 1.0,
                'structural_divergence': 0.0,
                'gramm_errors': 0,
                'gramm_errors_spelling': 0,
                'gramm_errors_grammar': 0,
                'gramm_errors_punctuation': 0,
                'local_semantic_window_size_1': 1.0,
                'local_semantic_window_size_2': 1.0,
                'local_semantic_window_size_3': 1.0,
                'global_semantic_cosine_similarity': 1.0,
                'global_semantic_coherence': 1.0,
                })
            continue

        try:
            scores               = shallow_bench(ref, hyp)
            lexical_score        = scores['lexical_fabrication_score']
            phonetic_score       = scores['phonetic_fabrication_score']
            morph_score          = scores['morphological_hallucination_score']
            local_semantic_score = scores['local_semantic_score']
            global_semantic_score = scores['global_semantic_score']
            all_results.append({
                'ref': ref,
                'hyp': hyp,
                'wer': scores['wer_score'],
                'ins_count': lexical_score['insertions_count'],
                'ins_ratio': lexical_score['insertions_ratio'],
                'del_count': lexical_score['deletions_count'],
                'del_ratio': lexical_score['deletions_ratio'],
                'sub_count': lexical_score['substitutions_count'],
                'sub_ratio': lexical_score['substitutions_ratio'],
                'phonetic_hamming': phonetic_score['hamming'],
                'phonetic_levenshtein': phonetic_score['levenshtein'],
                'phonetic_jaro_winkler': phonetic_score['jaro_winkler'],
                'structural_divergence': morph_score['structural_divergence'],
                'gramm_errors': morph_score['grammatical_errors']['total_errors'],
                'gramm_errors_spelling': morph_score['grammatical_errors']['error_categories']['spelling'],
                'gramm_errors_grammar': morph_score['grammatical_errors']['error_categories']['grammar'],
                'gramm_errors_punctuation': morph_score['grammatical_errors']['error_categories']['punctuation'],
                'local_semantic_window_size_1': local_semantic_score['window_size_1'],
                'local_semantic_window_size_2': local_semantic_score['window_size_2'],
                'local_semantic_window_size_3': local_semantic_score['window_size_3'],
                'global_semantic_cosine_similarity': global_semantic_score['global_semantic_cosine_similarity'],
                'global_semantic_coherence': global_semantic_score['global_semantic_coherence'],
                })
        except Exception as e:
            total_errors += 1
            print(f'Error on example {idx}: {str(e)}', flush=True)

    print(f'Processing complete. Total errors: {total_errors}', flush=True)

    if not all_results:
        print('No results were processed successfully. Exiting.', flush=True)
        return

    df = pd.DataFrame(all_results)

    if len(segment_ids) == len(df):
        df['segment_id'] = segment_ids
        df['participant'] = df['segment_id'].str.extract(r'DISCOURSE_(\d+)_')[0]
        df['task'] = df['segment_id'].str.replace(r'DISCOURSE_\d+_', '', regex=True)
    else:
        print(f'⚠ Segment ID mismatch: {len(segment_ids)} IDs vs {len(df)} results', flush=True)

    # -------------------------------------------------------------------------
    # Aggregated scoring
    # -------------------------------------------------------------------------

    print('Saving partial results', flush=True)
    df.to_csv(f'{args.output_dir}/shallow_metrics_{args.dataset_name}_{args.model_name}_partial.csv', index=False)

    print('Calculating WER for the entire dataset', flush=True)
    wer_score = shallow_bench.compute_dataset_wer(gt_transcriptions, pred_transcriptions)

    print('Calculating lexical fabrication scores', flush=True)
    lexical_fabrication_scores = shallow_bench.aggregated_lexical_fabrication_score(
        ins_ratios=df['ins_ratio'].tolist(),
        del_ratios=df['del_ratio'].tolist(),
        sub_ratios=df['sub_ratio'].tolist(),
        hypotheses=pred_transcriptions
        )
    df['lexical_fabrication_score'] = lexical_fabrication_scores

    print('Calculating phonetic fabrication scores', flush=True)
    phonetic_fabrication_scores = shallow_bench.aggregated_phonetic_score(
        hammings=df['phonetic_hamming'].tolist(),
        levenshteins=df['phonetic_levenshtein'].tolist(),
        jaro_winklers=df['phonetic_jaro_winkler'].tolist(),
        )
    df['phonetic_fabrication_score'] = phonetic_fabrication_scores

    print('Calculating morphological hallucination scores', flush=True)
    structural_divergence_scores, grammatical_errors_scores, morphological_hallucination_scores = \
        shallow_bench.aggregated_morphological_hallucination_score(
            syntax_divergences=df['structural_divergence'].tolist(),
            spelling_errors=df['gramm_errors_spelling'].tolist(),
            grammar_errors=df['gramm_errors_grammar'].tolist(),
            punctuation_errors=df['gramm_errors_punctuation'].tolist(),
            hypotheses=pred_transcriptions
            )
    df['structural_divergence_score']       = structural_divergence_scores
    df['grammatical_errors_score']          = grammatical_errors_scores
    df['morphological_hallucination_score'] = morphological_hallucination_scores

    print('Calculating contextual hallucination scores', flush=True)
    local_semantic_scores = shallow_bench.aggregated_local_semantic_score(
        c1s=df['local_semantic_window_size_1'].tolist(),
        c2s=df['local_semantic_window_size_2'].tolist(),
        c3s=df['local_semantic_window_size_3'].tolist(),
        )
    df['local_semantic_score'] = local_semantic_scores

    print('Calculating global semantic scores', flush=True)
    global_semantic_scores = shallow_bench.aggregated_global_semantic_score(
        cosines=df['global_semantic_cosine_similarity'].tolist(),
        semantic_coherences=df['global_semantic_coherence'].tolist(),
        )
    df['global_semantic_score'] = global_semantic_scores

    print('Calculating semantic hallucination scores', flush=True)
    semantic_score = shallow_bench.aggregated_semantic_score(
        local_semantic_scores=df['local_semantic_score'].tolist(),
        global_semantic_scores=df['global_semantic_score'].tolist()
        )
    df['semantic_hallucination_score'] = semantic_score

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    print('Calculating dataset statistics', flush=True)
    stats = shallow_bench.compute_dataset_stats(df)

    print('Saving results', flush=True)
    data = {
        'wer_score': wer_score,
        'lexical_fabrication_score': stats['lexical_fabrication_score'],
        'phonetic_fabrication_score': stats['phonetic_fabrication_score'],
        'morphological_hallucination_score': stats['morphological_hallucination_score'],
        'semantic_hallucination_score': stats['semantic_hallucination_score'],
        }
    with open(f'{args.output_dir}/shallow_stats_{args.dataset_name}_{args.model_name}.json', 'w') as f:
        json.dump(data, f, indent=4)

    print('Saving complete dataset to CSV', flush=True)
    df.to_csv(f'{args.output_dir}/shallow_metrics_{args.dataset_name}_{args.model_name}.csv', index=False)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()