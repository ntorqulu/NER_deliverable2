from __future__ import annotations

import os
import time
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from pathlib import Path
import random

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):
        return it


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, classification_report,
)

DATA_DIR  = Path('data')
MODEL_DIR = Path('fitted_models')

#############################################
# 1. DATA LOADING
#############################################

def load_ner_csv(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Load NER CSV (columns: sentence_id, words, tags)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    id_col, word_col, tag_col = "sentence_id", "words", "tags"
    df = df.dropna(subset=[word_col, tag_col])
    df[word_col] = df[word_col].astype(str)
    df[tag_col]  = df[tag_col].astype(str)
    sentences, labels = [], []
    for _, grp in df.groupby(id_col, sort=True):
        sentences.append(grp[word_col].tolist())
        labels.append(grp[tag_col].tolist())
    return sentences, labels


def load_tiny_test(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    return load_ner_csv(path)


#############################################
# 2. CRF FEATURE ENGINEERING  (improved)
#############################################

def word2features(sent: List[str], i: int) -> Dict[str, object]:
    """
    Rich feature set for CRF.
    Improvements over baseline:
      - 4-char suffixes/prefixes
      - word shape (Xxx, XXX, xxx, xdx …)
      - camelCase flag
      - all-caps ratio
      - extended context window features (±2)
      - bigram context combinations
    """
    word = sent[i]
    lw   = word.lower()

    def shape(w):
        s = re.sub(r'[A-Z]', 'X', w)
        s = re.sub(r'[a-z]', 'x', s)
        s = re.sub(r'[0-9]', 'd', s)
        return s

    features: Dict[str, object] = {
        # bias
        "bias": 1.0,
        # word identity
        "word.lower":    lw,
        # suffixes
        "word[-2:]":     lw[-2:],
        "word[-3:]":     lw[-3:],
        "word[-4:]":     lw[-4:],
        # prefixes
        "word[:2]":      lw[:2],
        "word[:3]":      lw[:3],
        "word[:4]":      lw[:4],
        # shape
        "word.shape":    shape(word),
        "word.shape3":   shape(word)[:3],
        # boolean flags
        "word.isupper":  word.isupper(),
        "word.istitle":  word.istitle(),
        "word.isdigit":  word.isdigit(),
        "word.hasdigit": any(c.isdigit() for c in word),
        "word.hashyphen": "-" in word,
        "word.hasdot":   "." in word,
        "word.is_camel": bool(re.search(r'[a-z][A-Z]', word)),
        "word.caps_ratio": round(sum(c.isupper() for c in word) / max(len(word), 1), 2),
        "word.len":      min(len(word), 20),   # binned length
    }

    # -2 context
    if i > 1:
        ppw = sent[i - 2]
        features.update({
            "-2:word.lower":  ppw.lower(),
            "-2:word.istitle": ppw.istitle(),
            "-2:word.isupper": ppw.isupper(),
        })

    # -1 context
    if i > 0:
        pw = sent[i - 1]
        features.update({
            "-1:word.lower":  pw.lower(),
            "-1:word.istitle": pw.istitle(),
            "-1:word.isupper": pw.isupper(),
            "-1:word.shape":  shape(pw),
            # bigram
            "-1:word.lower+word.lower": pw.lower() + "_" + lw,
        })
    else:
        features["BOS"] = True

    # +1 context
    if i < len(sent) - 1:
        nw = sent[i + 1]
        features.update({
            "+1:word.lower":  nw.lower(),
            "+1:word.istitle": nw.istitle(),
            "+1:word.isupper": nw.isupper(),
            "+1:word.shape":  shape(nw),
            # bigram
            "word.lower+1:word.lower": lw + "_" + nw.lower(),
        })
    else:
        features["EOS"] = True

    # +2 context
    if i < len(sent) - 2:
        nnw = sent[i + 2]
        features.update({
            "+2:word.lower":  nnw.lower(),
            "+2:word.istitle": nnw.istitle(),
            "+2:word.isupper": nnw.isupper(),
        })

    return features


def sent2features(sent: List[str]) -> List[Dict]:
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(labels: List[str]) -> List[str]:
    return list(labels)


def encode_crf(sentences: List[List[str]], labels: List[List[str]]):
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(l)   for l in labels]
    return X, y


#############################################
# 3. STRUCTURED PERCEPTRON  (improved)
#############################################

def sp_token_features(tokens: List[str], i: int, prev_label: str, label: str) -> List[str]:
    """
    Feature function for the Structured Perceptron.
    Improvements: 4-char affixes, shape, bigram context, camelCase flag.
    """
    word = tokens[i]

    def shape(w):
        s = re.sub(r'[A-Z]', 'X', w)
        s = re.sub(r'[a-z]', 'x', s)
        return re.sub(r'[0-9]', 'd', s)

    feats = [
        f"word={word.lower()}::{label}",
        f"suffix2={word[-2:].lower()}::{label}",
        f"suffix3={word[-3:].lower()}::{label}",
        f"suffix4={word[-4:].lower()}::{label}",
        f"prefix2={word[:2].lower()}::{label}",
        f"prefix3={word[:3].lower()}::{label}",
        f"prefix4={word[:4].lower()}::{label}",
        f"shape={shape(word)}::{label}",
        f"is_upper={word[0].isupper()}::{label}",
        f"is_all_upper={word.isupper()}::{label}",
        f"is_title={word.istitle()}::{label}",
        f"has_digit={any(c.isdigit() for c in word)}::{label}",
        f"has_hyphen={'-' in word}::{label}",
        f"is_camel={bool(re.search(r'[a-z][A-Z]', word))}::{label}",
        f"transition:{prev_label}->{label}",
    ]
    if i > 0:
        pw = tokens[i - 1]
        feats += [
            f"prev_word={pw.lower()}::{label}",
            f"prev_is_upper={pw[0].isupper()}::{label}",
            f"prev_shape={shape(pw)}::{label}",
            f"bigram={pw.lower()}_{word.lower()}::{label}",
        ]
    if i > 1:
        ppw = tokens[i - 2]
        feats.append(f"prev2_word={ppw.lower()}::{label}")

    if i < len(tokens) - 1:
        nw = tokens[i + 1]
        feats += [
            f"next_word={nw.lower()}::{label}",
            f"next_is_upper={nw[0].isupper()}::{label}",
            f"next_shape={shape(nw)}::{label}",
        ]
    else:
        feats.append(f"transition:{label}-><STOP>")

    if i < len(tokens) - 2:
        nnw = tokens[i + 2]
        feats.append(f"next2_word={nnw.lower()}::{label}")

    return feats


def sp_score(weights: defaultdict, feats: List[str]) -> float:
    return sum(weights[f] for f in feats)


def sp_viterbi(weights: defaultdict, tokens: List[str], label_set: List[str], feat_fn) -> List[str]:
    """
    Viterbi decoding for the Structured Perceptron.

    Speedup: for each position i and candidate label j, the feature score is
    split into two parts:
      - emit[i, j]: features that depend only on (position, current label) —
        computed once per (i, j).
      - trans[k, j]: features that depend on (prev label, current label) —
        specifically the transition feature 'k->j', computed once per (k, j)
        pair regardless of position.

    The transition matrix is precomputed once before the loop (it does not
    depend on position), reducing the per-token work from O(L^2) sp_score
    calls to O(L) sp_score calls.
    """
    n = len(tokens)
    L = len(label_set)

    # ── 1. Precompute transition scores (L x L matrix, position-independent) ──
    # trans[k, j] = weight of the feature "transition: label_k -> label_j"
    trans = np.zeros((L, L), dtype=np.float64)
    for k, prev in enumerate(label_set):
        for j, label in enumerate(label_set):
            trans[k, j] = weights[f"transition:{prev}->{label}"]

    # ── 2. Viterbi forward pass ──
    vit = np.full((n, L), -np.inf, dtype=np.float64)
    bp  = np.zeros((n, L), dtype=np.int32)

    # Position 0: use <START> as previous label
    for j, label in enumerate(label_set):
        feats = feat_fn(tokens, 0, "<START>", label)
        vit[0, j] = sp_score(weights, feats)

    for i in range(1, n):
        # Emission: features that depend on (position i, current label j)
        # but NOT on the previous label. We approximate by scoring feat_fn
        # once with a dummy prev and subtracting the transition feature.
        emit_i = np.zeros(L, dtype=np.float64)
        for j, label in enumerate(label_set):
            feats = feat_fn(tokens, i, label_set[0], label)
            # subtract the transition feature we added with dummy prev
            emit_i[j] = (sp_score(weights, feats)
                         - weights[f"transition:{label_set[0]}->{label}"])

        # scores[k, j] = vit[i-1, k] + trans[k, j] + emit_i[j]
        scores = vit[i - 1, :, None] + trans + emit_i[None, :]
        bp[i]  = np.argmax(scores, axis=0)
        vit[i] = scores[bp[i], np.arange(L)]

    # ── 3. Backtrack ──
    seq = [int(np.argmax(vit[n - 1]))]
    for i in range(n - 1, 0, -1):
        seq.append(int(bp[i, seq[-1]]))
    return [label_set[j] for j in reversed(seq)]


def sp_update(weights: defaultdict, tokens, y_true, y_pred, feat_fn):
    prev_true = prev_pred = "<START>"
    for i, (gold, pred) in enumerate(zip(y_true, y_pred)):
        if gold != pred or prev_true != prev_pred:
            for f in feat_fn(tokens, i, prev_true, gold):
                weights[f] += 1
            for f in feat_fn(tokens, i, prev_pred, pred):
                weights[f] -= 1
        prev_true = gold
        prev_pred = pred


def train_structured_perceptron(
    train_sents: List[List[str]],
    train_labels: List[List[str]],
    label_set: List[str],
    n_epochs: int = 10,
    seed: int = 42,
    averaged: bool = True,
) -> defaultdict:
    """
    Train an (Averaged) Structured Perceptron for NER.

    Averaging is done lazily: instead of accumulating the full weight vector
    after every sentence (O(|weights|) per step), we track for each feature
    the last step at which it changed and accumulate only at that point.
    This reduces the averaging overhead from O(|weights| * T) to O(updates),
    giving a ~10-50x speedup when the weight vector is large and sparse.
    """
    weights    = defaultdict(float)
    # lazy averaging: store (cumulative_sum, last_update_step) per feature
    cum        = defaultdict(float)   # cumulative sum
    last_t     = defaultdict(int)     # step at which feature was last updated
    t          = 0                    # global step counter
    indices    = list(range(len(train_sents)))
    random.seed(seed)
    total_start = time.time()
    n = len(train_sents)

    def _update_feature(f: str, delta: float) -> None:
        """Apply a weight update with lazy cumsum bookkeeping."""
        # flush the contribution of the current weight value up to now
        cum[f]    += weights[f] * (t - last_t[f])
        last_t[f]  = t
        weights[f] += delta

    epoch_bar = _tqdm(range(n_epochs), desc="SP training", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.time()
        random.shuffle(indices)
        errors = 0

        for idx in indices:
            y_pred = sp_viterbi(weights, train_sents[idx], label_set, sp_token_features)
            if y_pred != list(train_labels[idx]):
                # update: reward gold features, penalise predicted features
                prev_true = prev_pred = "<START>"
                for i, (gold, pred) in enumerate(zip(train_labels[idx], y_pred)):
                    if gold != pred or prev_true != prev_pred:
                        for f in sp_token_features(train_sents[idx], i, prev_true, gold):
                            _update_feature(f, +1.0)
                        for f in sp_token_features(train_sents[idx], i, prev_pred, pred):
                            _update_feature(f, -1.0)
                    prev_true = gold
                    prev_pred = pred
                errors += 1
            t += 1

        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - total_start
        remaining  = epoch_time * (n_epochs - epoch - 1)
        epoch_bar.set_postfix(
            errors=f"{errors}/{n}",
            time=f"{epoch_time:.0f}s",
            ETA=f"{remaining/60:.0f}min",
        )
        print(f"  Epoch {epoch+1:2d}/{n_epochs}: errors={errors}/{n}  "
              f"time={epoch_time:.0f}s  elapsed={elapsed/60:.1f}min  "
              f"ETA={remaining/60:.0f}min")

    total = time.time() - total_start
    print(f"\nTraining complete in {total/60:.1f} min")

    if not averaged:
        return weights

    # finalise lazy cumsum: flush any features not updated on the last step
    T = t
    for f in weights:
        cum[f] += weights[f] * (T - last_t[f])
    return defaultdict(float, {f: cum[f] / T for f in cum})


#############################################
# 4. BILSTM VOCAB HELPERS
#############################################

def build_vocab(sentences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    counts = Counter(w for s in sentences for w in s)
    vocab  = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counts.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def build_tag_map(labels: List[List[str]]) -> Dict[str, int]:
    tags = sorted({t for seq in labels for t in seq})
    return {t: i for i, t in enumerate(tags)}


def load_glove_embeddings(vocab: Dict[str, int], glove_path: str,
                           embed_dim: int = 100) -> "np.ndarray":
    """
    Load GloVe vectors from a local .txt file and align to vocab.
    Returns a (vocab_size, embed_dim) numpy array.
    """
    embed_matrix = np.random.normal(0, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    embed_matrix[0] = 0.0   # PAD stays zero

    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in vocab:
                embed_matrix[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"GloVe: {found}/{len(vocab)} vocab words found ({100*found/len(vocab):.1f}%)")
    return embed_matrix


#############################################
# 5. EVALUATION HELPERS
#############################################

def flatten_exclude_O(y_true_seqs, y_pred_seqs):
    yt, yp = [], []
    for true_seq, pred_seq in zip(y_true_seqs, y_pred_seqs):
        for t, p in zip(true_seq, pred_seq):
            if t != "O":
                yt.append(t)
                yp.append(p)
    return yt, yp


def evaluate_model(y_true_seqs, y_pred_seqs, label_names=None, exclude_O=True):
    if exclude_O:
        yt, yp = flatten_exclude_O(y_true_seqs, y_pred_seqs)
    else:
        yt = [label for seq in y_true_seqs for label in seq]
        yp = [label for seq in y_pred_seqs for label in seq]

    labels = sorted(set(yt))
    acc    = accuracy_score(yt, yp)
    f1_mac = f1_score(yt, yp, average="macro",    labels=labels, zero_division=0)
    f1_wt  = f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)
    report = classification_report(yt, yp, labels=labels, digits=4, zero_division=0)
    cm     = confusion_matrix(yt, yp, labels=labels)
    return {
        "accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt,
        "report": report, "cm": cm, "cm_labels": labels,
        "y_true_flat": yt, "y_pred_flat": yp,
    }


def print_tiny_test(sentences: List[List[str]], predictions: List[List[str]]):
    for sent, preds in zip(sentences, predictions):
        print(" ".join(f"{w}/{t}" for w, t in zip(sent, preds)))


def format_confusion_matrix(cm, labels):
    df = pd.DataFrame(cm,
                      index=[f"true:{l}" for l in labels],
                      columns=[f"pred:{l}" for l in labels])
    return df.to_string()


def full_eval(model_name, y_true_seqs, y_pred_seqs, split_name):
    res = evaluate_model(y_true_seqs, y_pred_seqs, exclude_O=True)
    print(f'\n{"="*60}')
    print(f'{model_name} — {split_name}  (non-O labels only)')
    print(f'  Accuracy   : {res["accuracy"]:.4f}')
    print(f'  F1 macro   : {res["f1_macro"]:.4f}')
    print(f'  F1 weighted: {res["f1_weighted"]:.4f}')
    print()
    print(res['report'])
    print('Confusion matrix:')
    print(format_confusion_matrix(res['cm'], res['cm_labels']))
    fig, ax = plt.subplots(figsize=(max(5, len(res['cm_labels'])),
                                    max(4, len(res['cm_labels']) - 1)))
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=res['cm_labels'], yticklabels=res['cm_labels'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{model_name} — {split_name} (non-O)')
    plt.tight_layout()
    fname = f'{model_name.lower().replace(" ", "_")}_{split_name.lower()}_cm.png'
    plt.savefig(MODEL_DIR / fname, dpi=120)
    plt.show()
    return res