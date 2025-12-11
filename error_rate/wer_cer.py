#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python wer_cer.py -i input_path.csv -o output_path.csv

from jiwer import wer as jiwer_wer
import Levenshtein
import editdistance
import argparse
import logging
import re
import unicodedata
import math
import pandas as pd

logging.basicConfig(
    format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s',
    level=logging.INFO
)


def levenshtein_with_ops(u, v):
    """
    Return (distance, (subs, ins, dels)) for sequences u and v.
    u, v can be strings (characters) or lists (tokens).
    """
    prev = None
    curr = [idx for idx in range(len(v) + 1)]

    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]  # (subs, dels, ins)

    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    dist = curr[len(v)]
    subs, dels, ins = curr_ops[len(v)]
    return dist, (subs, ins, dels)


def safe_div(num, den):
    return float('nan') if den == 0 else num / den


# ---------- NEW: light “original-style” normalization (no lowercasing) ----------
def normalize_for_original_metrics(s: str) -> str:
    # 1. flatten line breaks
    s = s.replace('\r', ' ').replace('\n', ' ')
    # 2. collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # 3. strip outer quotes
    s = s.strip('"').strip("'")
    # 4. drop trailing page-artifact tokens: bare roman numerals or digits
    tokens = s.split()
    tokens = [t for t in tokens if t != '—']
    tokens = [t for t in tokens if t != '-']
    while tokens:
        core = re.sub(r'^\W+|\W+$', '', tokens[-1])  # trim punctuation
        if re.fullmatch(r'[ivxlcdm]+|\d+', core, flags=re.IGNORECASE):
            tokens.pop()
        else:
            break
    return ' '.join(tokens)
# ------------------------------------------------------------------------------


def normalize_text(text):
    """
    Heavier normalization for *_norm metrics.
    This one DOES lowercase.
    """
    # fix unicode (quotes, dashes, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # remove weird OCR chars
    text = re.sub(r'[@^~•·§¤]', '', text)

    # collapse all whitespace (including newlines)
    text = re.sub(r'\s+', ' ', text)

    # remove space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # remove linebreak hyphens and join words
    text = re.sub(r'-\s*\n\s*', '', text)

    # lowercase for fair comparison
    text = text.lower()

    return text.strip()


def compute_metrics_for_pair(ref, hyp):
    """
    Compute CER/WER for a single ref/hyp pair.

    CER: distance / len(ref)
    WER: (S + I + D) / len(ref_words)
    """
    ref = ref or ""
    hyp = hyp or ""

    char_dist, (s_c, i_c, d_c) = levenshtein_with_ops(ref, hyp)
    denom_chars = len(ref)
    cer = safe_div(char_dist, denom_chars)

    ref_words = ref.split()
    hyp_words = hyp.split()
    word_dist, (s_w, i_w, d_w) = levenshtein_with_ops(ref_words, hyp_words)
    wer = safe_div(word_dist, len(ref_words))

    return cer, wer


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute CER/WER per row for a CSV with target_gsent, "
            "target_hsent, and matched_hsent."
        )
    )
    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help='Path to input CSV with columns target_gsent, target_hsent, matched_hsent'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV with added metric columns'
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, on_bad_lines="skip", engine="python")

    required_cols = ['target_gsent', 'target_hsent', 'matched_hsent']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV.")

    cer_gh_list = []       # gold vs hathi (original-style)
    wer_gh_list = []
    cer_gh_norm_list = []
    wer_gh_norm_list = []

    cer_gm_list = []       # gold vs matched
    wer_gm_list = []
    cer_gm_norm_list = []
    wer_gm_norm_list = []

    cer_hm_list = []       # hathi vs matched
    wer_hm_list = []
    cer_hm_norm_list = []
    wer_hm_norm_list = []

    for idx, row in df.iterrows():
        gsent = str(row.get('target_gsent', '') or '')
        hsent = str(row.get('target_hsent', '') or '')
        msent = str(row.get('matched_hsent', '') or '')

        # ---------- NEW: original-style normalization for main metrics ----------
        g_orig = normalize_for_original_metrics(gsent)
        h_orig = normalize_for_original_metrics(hsent)
        m_orig = normalize_for_original_metrics(msent)
        # -----------------------------------------------------------------------

        # --- 1. Gold vs Hathi (original-style, no lowercasing, just flattening) ---
        cer_gh, wer_gh = compute_metrics_for_pair(g_orig, h_orig)
        cer_gh_list.append(cer_gh)
        wer_gh_list.append(wer_gh)

        # --- 1b. Gold vs Hathi with heavier normalization (your *_norm) ---
        gsent_norm = normalize_text(gsent)
        hsent_norm = normalize_text(hsent)
        cer_gh_norm, wer_gh_norm = compute_metrics_for_pair(gsent_norm, hsent_norm)
        cer_gh_norm_list.append(cer_gh_norm)
        wer_gh_norm_list.append(wer_gh_norm)

        # --- 2. Gold vs Matched (should replicate original cer/wer columns) ---
        cer_gm, wer_gm = compute_metrics_for_pair(g_orig, m_orig)
        cer_gm_list.append(cer_gm)
        wer_gm_list.append(wer_gm)

        msent_norm = normalize_text(msent)
        cer_gm_norm, wer_gm_norm = compute_metrics_for_pair(gsent_norm, msent_norm)
        cer_gm_norm_list.append(cer_gm_norm)
        wer_gm_norm_list.append(wer_gm_norm)

        # --- 3. Hathi vs Matched ---
        cer_hm, wer_hm = compute_metrics_for_pair(h_orig, m_orig)
        cer_hm_list.append(cer_hm)
        wer_hm_list.append(wer_hm)

        hsent_norm = normalize_text(hsent)  # recompute or reuse above
        cer_hm_norm, wer_hm_norm = compute_metrics_for_pair(hsent_norm, msent_norm)
        cer_hm_norm_list.append(cer_hm_norm)
        wer_hm_norm_list.append(wer_hm_norm)

    # 1) gold vs Hathi
    df['cer_gh'] = cer_gh_list
    df['wer_gh'] = wer_gh_list
    df['cer_gh_norm'] = cer_gh_norm_list
    df['wer_gh_norm'] = wer_gh_norm_list

    # 2) gold vs matched
    df['cer_gm'] = cer_gm_list
    df['wer_gm'] = wer_gm_list
    df['cer_gm_norm'] = cer_gm_norm_list
    df['wer_gm_norm'] = wer_gm_norm_list

    # 3) hathi vs matched
    df['cer_hm'] = cer_hm_list
    df['wer_hm'] = wer_hm_list
    df['cer_hm_norm'] = cer_hm_norm_list
    df['wer_hm_norm'] = wer_hm_norm_list

    # Compare to existing original cer/wer columns (which are g vs matched)
    if 'cer' in df.columns:
        df['cer_diff_from_orig'] = df['cer_gh'] - df['cer']
    else:
        logging.warning("Original 'cer' column not found in input CSV.")

    if 'wer' in df.columns:
        df['wer_diff_from_orig'] = df['wer_gh'] - df['wer']
    else:
        logging.warning("Original 'wer' column not found in input CSV.")

    df.to_csv(args.output_csv, index=False)
    logging.info("Wrote updated CSV to %s", args.output_csv)


if __name__ == '__main__':
    main()
