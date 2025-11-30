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


def normalize_text(text):
    # fix unicode (quotes, dashes, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # remove weird OCR chars
    text = re.sub(r'[@^~•·§¤]', '', text)

    # collapse whitespace
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
    Compute CER/WER for raw ref/hyp strings.
    CER uses distance / len(ref).
    WER uses (S+I+D) / len(ref_words).
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
        description='Compute CER/WER per row for a CSV of gsent/hsent.'
    )
    parser.add_argument(
        '-i', '--input_csv',
        type=str,
        required=True,
        help='Path to input CSV with columns gsent and hsent'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV with added metric columns'
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    cer_raw_list = []
    wer_raw_list = []
    cer_norm_list = []
    wer_norm_list = []

    for idx, row in df.iterrows():
        gsent = str(row.get('gsent', '') or '')
        hsent = str(row.get('hsent', '') or '')

        cer_raw, wer_raw = compute_metrics_for_pair(gsent, hsent)

        gsent_norm = normalize_text(gsent)
        hsent_norm = normalize_text(hsent)
        cer_norm, wer_norm = compute_metrics_for_pair(gsent_norm, hsent_norm)

        cer_raw_list.append(cer_raw)
        wer_raw_list.append(wer_raw)
        cer_norm_list.append(cer_norm)
        wer_norm_list.append(wer_norm)

    df['cer_new_text'] = cer_raw_list
    df['wer_new_text'] = wer_raw_list
    df['cer_normalized_text'] = cer_norm_list
    df['wer_normalized_text'] = wer_norm_list


    if 'cer' in df.columns:
        df['cer_delta_new'] = df['cer_new_text'] - df['cer']
        df['cer_delta_norm'] = df['cer_normalized_text'] - df['cer']
    else:
        logging.warning("Original 'cer' column not found in input CSV.")

    if 'wer' in df.columns:
        df['wer_delta_new'] = df['wer_new_text'] - df['wer']
        df['wer_delta_norm'] = df['wer_normalized_text'] - df['wer']
    else:
        logging.warning("Original 'wer' column not found in input CSV.")

    df.to_csv(args.output_csv, index=False)
    logging.info("Wrote updated CSV to %s", args.output_csv)


if __name__ == '__main__':
    main()