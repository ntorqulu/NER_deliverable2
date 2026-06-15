from __future__ import annotations

import os
import re
from collections import Counter
from typing import List, Tuple, Dict
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report,
)

from collections import defaultdict

DATA_DIR  = Path('data')
MODEL_DIR = Path('fitted_models')

#############################################
# 1. DATA LOADING
#############################################


def load_ner_csv(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load NER csv tagged data (columns: sentence_id, word, tag)

    :path: path to the csv file
    :dtype: str
    :return: (sentences, labels) where each element is a list of strings
    :rtype: Tuple[List[List[str]], List[List[str]]]
    """
    df = pd.read_csv(path)
    # Normalize the column names
    df.columns = [c.strip().lower() for c in df.columns]

    id_col = "sentence_id"
    word_col = "words"
    tag_col = "tags"

    # Drop rows where word or tag is NaN
    df = df.dropna(subset=[word_col, tag_col])
    df[word_col] = df[word_col].astype(str)
    df[tag_col] = df[tag_col].astype(str)

    sentences, labels = [], []
    for _, grp in df.groupby(id_col, sort=True):
        sentences.append(grp[word_col].astype(str).tolist())
        labels.append(grp[tag_col].astype(str).tolist())
    return sentences, labels


def load_tiny_test(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load tiny_test.csv tagged data (columns: sentence_id, word, tag)

    :path: path to the csv file
    :dtype: str
    :return: (sentences, labels) where each element is a list of strings
    :rtype: Tuple[List[List[str]], List[List[str]]]
    """
    return load_ner_csv(path)


#############################################
# 2. CRF FEATURE ENGINEERING
#############################################


def word2features(sent: List[str], i: int) -> Dict[str, object]:
    """
    Feature set for the CRF model at position i.

    :sent: list of words in the sentence
    :dtype: List[str]
    :i: index of the word in the sentence
    :dtype: int
    :return: dictionary of features for the word at position i
    :rtype: Dict[str, object]
    """
    word = sent[i]
    lw = word.lower()

    features: Dict[str, object] = {
        "bias": 1.0,
        "word.lower": lw,
        "word[-3:]": lw[-3:],
        "word[-2:]": lw[-2:],
        "word[:3]": lw[:3],
        "word[:2]": lw[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.hasdigit": any(c.isdigit() for c in word),
        "word.hashyphen": "-" in word,
        "word.hasdot": "." in word,
        "word.len": len(word),
    }

    if i > 0:
        pw = sent[i - 1]
        features.update(
            {
                "-1:word.lower": pw.lower(),
                "-1:word.istitle": pw.istitle(),
                "-1:word.isupper": pw.isupper(),
            }
        )
    else:
        features["BOS"] = True

    if i > 1:
        ppw = sent[i - 2]
        features.update(
            {
                "-2:word.lower": ppw.lower(),
                "-2:word.istitle": ppw.istitle(),
            }
        )

    if i < len(sent) - 1:
        nw = sent[i + 1]
        features.update(
            {
                "+1:word.lower": nw.lower(),
                "+1:word.istitle": nw.istitle(),
                "+1:word.isupper": nw.isupper(),
            }
        )
    else:
        features["EOS"] = True

    if i < len(sent) - 2:
        nnw = sent[i + 2]
        features.update(
            {
                "+2:word.lower": nnw.lower(),
                "+2:word.istitle": nnw.istitle(),
            }
        )

    return features


def sent2features(sent: List[str]) -> List[Dict]:
    """
    Convert a sentence into a list of feature dictionaries for each word.

    :sent: list of words in the sentence
    :dtype: List[str]
    :return: list of feature dictionaries for each word in the sentence
    :rtype: List[Dict]
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(labels: List[str]) -> List[str]:
    """
    Convert a list of labels into a list of strings.

    :labels: list of labels for the sentence
    :dtype: List[str]
    :return: list of labels as strings
    :rtype: List[str]
    """
    return list(labels)


def encode_crf(sentences: List[List[str]], labels: List[List[str]]):
    """
    Encode sentences and labels into features and labels for CRF model training.

    :sentences: list of sentences, where each sentence is a list of words
    :dtype: List[List[str]]
    :labels: list of labels, where each label is a list of strings corresponding to the words in the sentence
    :dtype: List[List[str]]
    :return: tuple of (X, y) where X is a list of feature dictionaries and y is a list of labels
    :rtype: Tuple[List[List[Dict]], List[List[str]]]
    """
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(l) for l in labels]
    return X, y


#############################################
# 3. Structured Perceptron feature functions.
# Reused from the notebook in the campus
#############################################


def sp_token_features(
    tokens: List[str], i: int, prev_label: str, label: str
) -> List[str]:
    """
    Feature set for the Structured Perceptron model at position i.

    :tokens: list of words in the sentence
    :dtype: List[str]
    :i: index of the word in the sentence
    :dtype: int
    :prev_label: label of the previous word
    :dtype: str
    :label: label of the current word
    :dtype: str
    :return: list of features for the word at position i
    :rtype: List[str]
    """
    word = tokens[i]
    feats = [
        f"word={word.lower()}::{label}",
        f"suffix2={word[-2:].lower()}::{label}",
        f"suffix3={word[-3:].lower()}::{label}",
        f"prefix2={word[:2].lower()}::{label}",
        f"prefix3={word[:3].lower()}::{label}",
        f"is_upper={word[0].isupper()}::{label}",
        f"is_all_upper={word.isupper()}::{label}",
        f"has_digit={any(c.isdigit() for c in word)}::{label}",
        f"has_hyphen={'-' in word}::{label}",
        f"transition:{prev_label}->{label}",
    ]
    if i > 0:
        pw = tokens[i - 1]
        feats += [
            f"prev_word={pw.lower()}::{label}",
            f"prev_is_upper={pw[0].isupper()}::{label}",
        ]
    if i < len(tokens) - 1:
        nw = tokens[i + 1]
        feats += [
            f"next_word={nw.lower()}::{label}",
            f"next_is_upper={nw[0].isupper()}::{label}",
        ]
    else:
        feats.append(f"transition:{label}-><STOP>")
    return feats


def sp_score(weights: defaultdict, feats: List[str]) -> float:
    """
    Compute the score for a given set of features using the provided weights.

    :weights: dictionary of feature weights
    :dtype: defaultdict
    :feats: list of features for the word at position i
    :dtype: List[str]
    :return: score for the given features
    :rtype: float
    """
    return sum(weights[f] for f in feats)


def sp_viterbi(
    weights: defaultdict, tokens: List[str], label_set: List[str], feat_fn
) -> List[str]:
    """
    Viterbi algorithm for decoding the best sequence of labels for a given sequence of tokens.

    :weights: dictionary of feature weights
    :dtype: defaultdict
    :tokens: list of words in the sentence
    :dtype: List[str]
    :label_set: list of possible labels
    :dtype: List[str]
    :feat_fn: feature function to compute features for each token
    :dtype: callable
    :return: best sequence of labels for the given tokens
    :rtype: List[str]
    """
    n = len(tokens)
    vit = [{}]
    bp = [{}]
    for label in label_set:
        vit[0][label] = sp_score(weights, feat_fn(tokens, 0, "<START>", label))
        bp[0][label] = "<START>"
    for i in range(1, n):
        vit.append({})
        bp.append({})
        for label in label_set:
            best_prev, best_s = max(
                (
                    (p, vit[i - 1][p] + sp_score(weights, feat_fn(tokens, i, p, label)))
                    for p in label_set
                ),
                key=lambda x: x[1],
            )
            vit[i][label] = best_s
            bp[i][label] = best_prev
    best_last = max(label_set, key=lambda y: vit[n - 1][y])
    seq = [best_last]
    for i in range(n - 1, 0, -1):
        seq.append(bp[i][seq[-1]])
    return list(reversed(seq))


def sp_update(weights: defaultdict, tokens, y_true, y_pred, feat_fn):
    """
    Update the weights of the features based on the true and predicted labels.

    :weights: dictionary of feature weights
    :dtype: defaultdict
    :tokens: list of words in the sentence
    :dtype: List[str]
    :y_true: list of true labels for the sentence
    :dtype: List[str]
    :y_pred: list of predicted labels for the sentence
    :dtype: List[str]
    :feat_fn: feature function to compute features for each token
    :dtype: callable
    """
    prev_true = prev_pred = "<START>"
    for i, (gold, pred) in enumerate(zip(y_true, y_pred)):
        if gold != pred or prev_true != prev_pred:
            for f in feat_fn(tokens, i, prev_true, gold):
                weights[f] += 1
            for f in feat_fn(tokens, i, prev_pred, pred):
                weights[f] -= 1
        prev_true = gold
        prev_pred = pred


def train_structured_perceptron(train_sents, train_labels, label_set,
                                 n_epochs=20, seed=42, averaged=True):
    weights  = defaultdict(float)
    cumsum   = defaultdict(float)   # running sum for averaging
    t        = 0                    # total update count
    indices  = list(range(len(train_sents)))
    random.seed(seed)

    for epoch in range(n_epochs):
        random.shuffle(indices)
        errors = 0
        for i in indices:
            y_pred = sp_viterbi(weights, train_sents[i], label_set, sp_token_features)
            if y_pred != list(train_labels[i]):
                sp_update(weights, train_sents[i], train_labels[i], y_pred, sp_token_features)
                errors += 1
            t += 1
            if averaged:
                for f, v in weights.items():
                    cumsum[f] += v
        print(f'  Epoch {epoch+1:2d}: sentence errors = {errors}/{len(train_sents)}')

    if averaged:
        final = defaultdict(float, {f: cumsum[f] / t for f in cumsum})
        return final
    return weights


#############################################
# 4. BILSTM FEATURE ENGINEERING
#############################################


def build_vocab(sentences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """
    Build a vocabulary from the given sentences, filtering out words below a minimum frequency.

    :sentences: list of sentences, where each sentence is a list of words
    :dtype: List[List[str]]
    :min_freq: minimum frequency for a word to be included in the vocabulary
    :dtype: int
    :return: dictionary mapping words to their indices in the vocabulary
    :rtype: Dict[str, int]
    """
    counts = Counter(w for s in sentences for w in s)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counts.most_common():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def build_tag_map(labels: List[List[str]]) -> Dict[str, int]:
    """
    Build a mapping from tags to indices based on the given labels.

    :labels: list of labels, where each label is a list of strings corresponding to the words in the sentence
    :dtype: List[List[str]]
    :return: dictionary mapping tags to their indices
    :rtype: Dict[str, int]
    """
    tags = sorted({t for seq in labels for t in seq})
    return {t: i for i, t in enumerate(tags)}


###############################################
# 5. EVALUATION HELPERS
###############################################


def flatten_exclude_O(y_true_seqs, y_pred_seqs):
    """
    Flatten the true and predicted label sequences, excluding the 'O' labels.

    :y_true_seqs: list of true label sequences
    :dtype: List[List[str]]
    :y_pred_seqs: list of predicted label sequences
    :dtype: List[List[str]]
    :return: tuple of (flattened true labels, flattened predicted labels) excluding 'O' labels
    :rtype: Tuple[List[str], List[str]]
    """
    yt, yp = [], []
    for true_seq, pred_seq in zip(y_true_seqs, y_pred_seqs):
        for t, p in zip(true_seq, pred_seq):
            if t != "O":
                yt.append(t)
                yp.append(p)
    return yt, yp


def evaluate_model(y_true_seqs, y_pred_seqs, label_names=None, exclude_O=True):
    """
    Evaluate the model's performance using accuracy, F1 score, and classification report.

    :y_true_seqs: list of true label sequences
    :dtype: List[List[str]]
    :y_pred_seqs: list of predicted label sequences
    :dtype: List[List[str]]
    :label_names: list of label names for classification report (optional)
    :dtype: List[str] or None
    :exclude_O: whether to exclude 'O' labels from evaluation
    :dtype: bool
    :return: dictionary containing accuracy, F1 score, and classification report
    :rtype: Dict[str, object]
    """
    if exclude_O:
        yt, yp = flatten_exclude_O(y_true_seqs, y_pred_seqs)
    else:
        yt = [label for seq in y_true_seqs for label in seq]
        yp = [label for seq in y_pred_seqs for label in seq]

    labels = sorted(set(yt))
    acc = accuracy_score(yt, yp)
    f1_mac = f1_score(yt, yp, average="macro", labels=labels, zero_division=0)
    f1_wt = f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)
    report = classification_report(yt, yp, labels=labels, digits=4, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "report": report,
        "cm": cm,
        "cm_labels": labels,
        "y_true_flat": yt,
        "y_pred_flat": yp,
    }


def print_tiny_test(sentences: List[List[str]], predictions: List[List[str]]):
    """
    Print the sentences and their corresponding predictions for a tiny test set.

    :sentences: list of sentences, where each sentence is a list of words
    :dtype: List[List[str]]
    :predictions: list of predicted label sequences, where each sequence is a list of strings corresponding to the words in the sentence
    :dtype: List[List[str]]
    """
    for sent, preds in zip(sentences, predictions):
        print(" ".join(f"{w}/{t}" for w, t in zip(sent, preds)))


def format_confusion_matrix(cm, labels):
    """
    Format the confusion matrix for display.

    :cm: confusion matrix as a 2D array
    :dtype: np.ndarray
    :labels: list of label names corresponding to the confusion matrix
    :dtype: List[str]
    :return: formatted string representation of the confusion matrix
    :rtype: str
    """
    df = pd.DataFrame(
        cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels]
    )
    return df.to_string()


def full_eval(model_name, y_true_seqs, y_pred_seqs, split_name):
    res = evaluate_model(y_true_seqs, y_pred_seqs, exclude_O=True)
    print(f'{"="*40}\nFull evaluation for {model_name} on {split_name} (non-O labels only)\n{"="*40}')
    print(f'{model_name} — {split_name}  (non-O labels only)')
    print(f'  Accuracy  : {res["accuracy"]:.4f}')
    print(f'  F1 macro  : {res["f1_macro"]:.4f}')
    print(f'  F1 weighted: {res["f1_weighted"]:.4f}')
    print()
    print(res['report'])
    # Confusion matrix
    print('Confusion matrix:')
    print(format_confusion_matrix(res['cm'], res['cm_labels']))
    # Plot CM
    fig, ax = plt.subplots(figsize=(max(5, len(res['cm_labels'])), max(4, len(res['cm_labels'])-1)))
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=res['cm_labels'], yticklabels=res['cm_labels'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{model_name} — {split_name} confusion matrix (non-O)')
    plt.tight_layout()
    fname = f'{model_name.lower().replace(" ","_")}_{split_name.lower()}_cm.png'
    plt.savefig(MODEL_DIR / fname, dpi=120)
    plt.show()
    return res
