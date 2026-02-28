#!/usr/bin/env python3

# This code is part of the SHALLOW benchmark, which aims to analyze hallucinations in ASR systems.
# The code is licensed under the Apache License CC-BY-SA 4.0
# You may not use this file except in compliance with the License.

# Script: semantic.py
# Modified by: Nadine El-Mufti (2026)
# Perf fix: cache window embeddings to avoid O(N^2) BERT forward passes per example.

from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import evaluate
import torch


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_pipeline_device(device_str):
    """
    Convert a torch device string to the integer index expected by the
    HuggingFace pipeline() API.
    """
    if device_str == 'cuda' or device_str.startswith('cuda:'):
        return int(device_str.split(':')[1]) if ':' in device_str else 0
    return -1


# -----------------------------------------------------------------------------
# SemanticAnalyzer
# -----------------------------------------------------------------------------

class SemanticAnalyzer:

    def __init__(
        self,
        local_model_name='bert-base-uncased',
        global_model_name='nli-roberta-base-v2',
        nli_model_name='facebook/bart-large-mnli',
        device='cuda'
        ):
        self.device       = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_tokenizer = AutoTokenizer.from_pretrained(local_model_name)
        self.local_model     = AutoModel.from_pretrained(local_model_name).to(self.device)
        self.global_model    = SentenceTransformer(global_model_name).to(self.device)

        pipeline_device = _resolve_pipeline_device(self.device)
        try:
            self.nli_model = pipeline('text-classification', model=nli_model_name, device=pipeline_device)
        except Exception:
            print(f'  Warning: NLI pipeline failed on device \'{self.device}\', falling back to CPU')
            self.nli_model = pipeline('text-classification', model=nli_model_name, device=-1)

        self._bertscore_metric = evaluate.load('bertscore', keep_in_memory=True)
        # Embedding cache: text -> numpy vector (avoids recomputing the same window text)
        self._emb_cache = {}

    # -------------------------------------------------------------------------
    # BERTScore + NLI scoring
    # -------------------------------------------------------------------------

    def _compute_semantic_similarity(self, embeddings):
        with torch.no_grad():
            cosine_sim = util.pytorch_cos_sim(embeddings[0].cpu(), embeddings[1].cpu())
        return cosine_sim.item()

    def _bertnli_semantic_score(self, reference, hypothesis):
        bs      = self._bertscore_metric.compute(predictions=[hypothesis], references=[reference], lang='en')
        bert_f1 = bs['f1'][0]

        nli_input  = f'{reference} </s> {hypothesis}'
        nli_result = self.nli_model(nli_input, truncation=True)[0]
        label      = nli_result['label']
        score      = nli_result['score']

        if label.lower() == 'entailment':
            entailment_prob = score
        elif label.lower() == 'neutral':
            entailment_prob = score * 0.5
        else:
            entailment_prob = 0.0

        final_score = bert_f1 * entailment_prob
        return round(final_score, 4), bert_f1, label, round(entailment_prob, 4)

    def _measure_semantic_coherence(self, reference, hypothesis):
        if reference == hypothesis:
            return 1.0
        elif reference.strip() == '' or hypothesis.strip() == '':
            return 0.0
        try:
            score, _, _, _ = self._bertnli_semantic_score(str(reference), str(hypothesis))
        except Exception as e:
            print(f'Error: {e}')
            score = 0.0
        return score

    # -------------------------------------------------------------------------
    # Local semantic scoring with embedding cache
    # -------------------------------------------------------------------------

    def _get_local_context_embedding(self, text):
        """
        Return the mean-pooled BERT embedding for text.
        Results are cached to avoid redundant forward passes for repeated window text.
        Cache is bounded to ~2000 entries to prevent unbounded memory growth.
        """
        if text in self._emb_cache:
            return self._emb_cache[text]

        inputs = self.local_tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.local_model(**inputs.to(self.device))
        vec = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()

        self._emb_cache[text] = vec
        if len(self._emb_cache) > 2000:
            keys = list(self._emb_cache.keys())
            for k in keys[:1000]:
                del self._emb_cache[k]
        return vec

    def _compute_local_semantic_coherence(self, ref_window, hyp_window):
        ref_vec = self._get_local_context_embedding(' '.join(ref_window))
        hyp_vec = self._get_local_context_embedding(' '.join(hyp_window))
        denom   = np.linalg.norm(ref_vec) * np.linalg.norm(hyp_vec)
        if denom == 0:
            return 0.0
        return float(np.dot(ref_vec, hyp_vec) / denom)

    def local_semantic_score(self, reference, hypothesis, window_sizes=[1, 2, 3]):
        if reference == hypothesis:
            return {f'window_size_{w}': 1.0 for w in window_sizes}
        elif reference.strip() == '' or hypothesis.strip() == '':
            return {f'window_size_{w}': 0.0 for w in window_sizes}

        r_words = reference.lower().split()
        h_words = hypothesis.lower().split()

        local_semantic_scores = {}
        for window_size in window_sizes:
            r_windows = [' '.join(r_words[j:j + window_size])
                         for j in range(len(r_words) - window_size + 1)]
            h_windows = [' '.join(h_words[i:i + window_size])
                         for i in range(len(h_words) - window_size + 1)]

            if not r_windows or not h_windows:
                local_semantic_scores[f'window_size_{window_size}'] = 0.0
                continue

            # Warm cache with all unique window texts
            for t in set(r_windows + h_windows):
                self._get_local_context_embedding(t)

            # Build normalized embedding matrices
            r_vecs  = np.stack([self._get_local_context_embedding(t) for t in r_windows])
            h_vecs  = np.stack([self._get_local_context_embedding(t) for t in h_windows])
            r_norms = np.linalg.norm(r_vecs, axis=1, keepdims=True)
            h_norms = np.linalg.norm(h_vecs, axis=1, keepdims=True)
            r_norms[r_norms == 0] = 1
            h_norms[h_norms == 0] = 1
            r_vecs_n = r_vecs / r_norms
            h_vecs_n = h_vecs / h_norms

            # For each hyp window, best cosine match over all ref windows
            sim_matrix  = h_vecs_n @ r_vecs_n.T       # (H, R)
            best_scores = sim_matrix.max(axis=1)       # (H,)
            score       = best_scores.sum() / max(len(r_words), len(h_words))
            local_semantic_scores[f'window_size_{window_size}'] = float(score)

        return local_semantic_scores

    # -------------------------------------------------------------------------
    # Global semantic scoring
    # -------------------------------------------------------------------------

    def global_semantic_score(self, reference, hypothesis):
        if reference == hypothesis:
            return {
                'global_semantic_cosine_similarity': 1.0,
                'global_semantic_coherence':         1.0
                }
        elif reference.strip() == '' or hypothesis.strip() == '':
            return {
                'global_semantic_cosine_similarity': 0.0,
                'global_semantic_coherence':         0.0
                }

        with torch.no_grad():
            embeddings = self.global_model.encode(
                [reference, hypothesis],
                convert_to_tensor=True
                )

        return {
            'global_semantic_cosine_similarity': self._compute_semantic_similarity(embeddings),
            'global_semantic_coherence':         self._measure_semantic_coherence(reference, hypothesis),
            }