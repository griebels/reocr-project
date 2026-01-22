#!/usr/bin/env python3
"""
Code to align Hathi OCR'd sentences to target sentences in a csv.

"""

from pathlib import Path
from functools import lru_cache
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz.distance import Levenshtein


# Data imports here.
TXT_DIR = Path("346_highlowpairs_hathi_vols_concat")  # folder with htid.txt files.
CSV_PATH = "difficult_matches_ooc.csv"               # input csv with hsent, gsent, hid
OUTPUT_PATH = "difficult_matches_matched5.csv"       # output path


# Helper function
def normalize_text(s: str) -> str:
    """
    Some fast and loose normalization for quick matching:
    - Unicode normalize
    - lowercase everyting
    - keep only letters, digits, spaces
    - collapse multiple spaces
    """
    if not isinstance(s, str):
        return ""

    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Load the txt files, split on sentences.
@lru_cache(maxsize=None)
def load_book_sentences(hid: str):
    """
    For a given hid, load the OCR'd txt file and split into sentences.

    Returns:
        sentences_orig: list[str]  # raw OCR sentences
        sentences_norm: list[str]  # normalized for matching
    """
    txt_path = TXT_DIR / f"{hid}.txt"
    if not txt_path.exists():
        print(f"WARNING: missing txt for {hid} at {txt_path}")
        return [], []

    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    sentences_orig = sent_tokenize(text)
    sentences_norm = [normalize_text(s) for s in sentences_orig]
    return sentences_orig, sentences_norm


# calculates how much from query sentence is contained in candidate sentence. (Rough guide)
def score_candidate_containment(query_norm: str, cand_norm: str) -> tuple[float, float]:
    """
    containment = |query_words ∩ cand_words| / |query_words|
    combined    = containment * 100 (0–100 scale)
    """
    if not query_norm or not cand_norm:
        return 0.0, 0.0

    q_words = query_norm.split()
    c_words = cand_norm.split()
    if not q_words or not c_words:
        return 0.0, 0.0

    q_set = set(q_words)
    c_set = set(c_words)

    containment = len(q_set & c_set) / len(q_set)
    combined = containment * 100.0
    return containment, combined


def anchor_filter_indices(hsent_norm: str, cand_norm_list: list[str], min_len: int = 5) -> list[int]:
    """
    Simple RETAS-style filter:
    - take the middle word of hsent_norm
    - keep only candidate indices where that word appears as a whole token
    """
    words = hsent_norm.split()
    if not words:
        return list(range(len(cand_norm_list)))

    anchor = words[len(words) // 2]
    if len(anchor) < min_len:
        return list(range(len(cand_norm_list)))

    anchor_indices = [i for i, cand in enumerate(cand_norm_list) if anchor in cand.split()]
    return anchor_indices if anchor_indices else list(range(len(cand_norm_list)))


# Honing in on the exact part of the candidate sentence(s) we want.
# This produces matched_honed_original --> in the end, this didn't 
# produce the best match, but keeping it for posterity.
def hone_by_average_length(
    hsent: str,
    gsent: str,
    matched_sentence: str,
    anchor_chars: int = 5,
):
    """
    Produce a shorter span from matched_sentence:
      - find an anchor (first 5 characters of hsent) in a newline-stripped copy
      - take avg_len characters, where avg_len = avg(len(hsent), len(gsent))
      - starting with anchor, take all characters up to avg_len
      - map back to original indices (preserving newlines)
      - extend to a word boundary (so it doesn't stop in the middle of a word).
    """
    if not isinstance(hsent, str) or not isinstance(gsent, str) or not isinstance(matched_sentence, str):
        return None

    hsent_strip = hsent.strip()
    gsent_strip = gsent.strip()
    if not hsent_strip or not gsent_strip or not matched_sentence:
        return None

    original_ms = matched_sentence

    # Build newline-free search string + index map back to original
    search_chars = []
    idx_map = []  # idx_map[i] -> index in original_ms
    for i, ch in enumerate(original_ms):
        if ch in ("\n", "\r"):
            continue
        search_chars.append(ch)
        idx_map.append(i)

    search_ms = "".join(search_chars)
    if not search_ms:
        return None

    anchor = hsent_strip if len(hsent_strip) < anchor_chars else hsent_strip[:anchor_chars]
    start_s = search_ms.lower().find(anchor.lower())
    if start_s == -1:
        return None

    avg_len = int(round((len(hsent_strip) + len(gsent_strip)) / 2))
    if avg_len <= 0:
        return None

    end_s = min(len(search_ms), start_s + avg_len)
    if end_s <= start_s:
        return None

    # Map back to original indices
    start_orig = idx_map[start_s]
    end_orig = idx_map[end_s - 1] + 1  # exclusive

    # Extend to word boundary in ORIGINAL
    while end_orig < len(original_ms):
        ch = original_ms[end_orig]
        if ch.isspace() or ch in ".,;:!?)]}\"'":
            break
        end_orig += 1

    return original_ms[start_orig:end_orig]


# This function produced matched_honed_lev.
# This was the better function to produce the final matched sentence.
def hone_by_levenshtein(
    hsent: str,
    matched_sentence: str,
    max_window_expand: int = 10,
    min_similarity: float = 0.7,
):
    """
    Find the best-matching substring of "matched_sentence" relative to "hsent"
    using RapidFuzz Levenshtein normalized similarity.

    Returns: (span, similarity) where similarity is between 0-1.
    """
    if not isinstance(hsent, str) or not isinstance(matched_sentence, str):
        return None, None

    pattern = hsent.strip()
    ms = matched_sentence
    if not pattern or not ms:
        return None, None

    pattern_l = pattern.lower()
    ms_l = ms.lower()

    Lp = len(pattern_l)
    n = len(ms_l)
    if Lp == 0 or n == 0:
        return None, None

    min_len = max(1, Lp - max_window_expand)
    max_len = min(n, Lp + max_window_expand)

    best_score = None
    best_start = None
    best_end = None

    score_cutoff = min_similarity

    for window_len in range(min_len, max_len + 1):
        if window_len > n:
            continue
        for start in range(0, n - window_len + 1):
            end = start + window_len
            cand = ms_l[start:end]

            score = Levenshtein.normalized_similarity(
                pattern_l,
                cand,
                score_cutoff=score_cutoff,
            )

            if score < score_cutoff:
                continue

            if (best_score is None) or (score > best_score):
                best_score = score
                best_start = start
                best_end = end
                score_cutoff = max(score_cutoff, score * 0.95)

    if best_score is None or best_score < min_similarity:
        return None, best_score

    start_orig = best_start
    end_orig = best_end

    # Extend to word boundary in ORIGINAL
    while end_orig < len(ms):
        ch = ms[end_orig]
        if ch.isspace() or ch in ".,;:!?)]}\"'":
            break
        end_orig += 1

    return ms[start_orig:end_orig], best_score


# ========= WORKER FUNCTION (runs in each process) =========
def process_one_hid(args):
    """
    Process all rows for a single hid (volume).
    Returns list of dicts with row_index + match fields.
    """
    hid, group = args
    matches = []

    sentences_orig, sentences_norm = load_book_sentences(hid)

    if not sentences_orig:
        for row_idx, _row in group.iterrows():
            matches.append(
                {
                    "row_index": row_idx,
                    "matched_sentence": None,
                    "matched_sent_idx": None,
                    "score_containment": 0.0,
                }
            )
        return matches

    # thresholds - these could be tweaked in future.
    MIN_CONTAINMENT = 0.20  # at least 20% of query words must appear
    MIN_COMBINED = 30.0     # minimum to accept a match at all

    n_sent = len(sentences_orig)

    for row_idx, row in group.iterrows():
        hsent_norm = row["hsent_norm"]

        if not sentences_norm:
            matches.append(
                {
                    "row_index": row_idx,
                    "matched_sentence": None,
                    "matched_sent_idx": None,
                    "score_containment": 0.0,
                }
            )
            continue

        candidate_indices = anchor_filter_indices(hsent_norm, sentences_norm)

        # ---- FIRST PASS: best single sentence ----
        best_idx = None
        best_combined = -1.0

        for i in candidate_indices:
            containment, combined = score_candidate_containment(hsent_norm, sentences_norm[i])

            if containment < MIN_CONTAINMENT:
                continue

            if combined > best_combined:
                best_idx = i
                best_combined = combined

        if best_idx is None or best_combined < MIN_COMBINED:
            matches.append(
                {
                    "row_index": row_idx,
                    "matched_sentence": None,
                    "matched_sent_idx": None,
                    "score_containment": 0.0,
                }
            )
            continue

        # ---- SECOND PASS: expand around center with neighbors ----
        center = best_idx  # already global since we searched the whole book

        windows = []
        windows.append([center])
        if center - 1 >= 0:
            windows.append([center - 1, center])
        if center + 1 < n_sent:
            windows.append([center, center + 1])
        if center - 1 >= 0 and center + 1 < n_sent:
            windows.append([center - 1, center, center + 1])

        best_window_indices = None
        best_window_score = -1.0
        best_window_containment = 0.0

        for idx_list in windows:
            window_norm = " ".join(sentences_norm[i] for i in idx_list)
            containment, combined = score_candidate_containment(hsent_norm, window_norm)
            if combined > best_window_score:
                best_window_score = combined
                best_window_indices = idx_list
                best_window_containment = containment

        if best_window_indices is None or best_window_score < MIN_COMBINED:
            matches.append(
                {
                    "row_index": row_idx,
                    "matched_sentence": None,
                    "matched_sent_idx": None,
                    "score_containment": 0.0,
                }
            )
            continue

        matched_sentence = " ".join(sentences_orig[i] for i in best_window_indices)

        matches.append(
            {
                "row_index": row_idx,
                "matched_sentence": matched_sentence,
                "matched_sent_idx": center,  # keep for debugging/analysis
                "score_containment": best_window_containment,
            }
        )

    return matches


# ========= MAIN PIPELINE =========
def main():
    nltk.download("punkt", quiet=True)

    df = pd.read_csv(CSV_PATH)

    # Normalize query sentence
    df["hsent_norm"] = df["hsent"].apply(normalize_text)

    # Group by volume id
    groups = list(df.groupby("hid"))

    all_matches = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_one_hid, (hid, group)) for hid, group in groups]
        for fut in as_completed(futures):
            all_matches.extend(fut.result())

    matches_df = pd.DataFrame(all_matches).set_index("row_index")

    # Join match results back to df
    df_out = df.join(
        matches_df[["matched_sentence", "matched_sent_idx", "score_containment"]],
        how="left",
    )

    # Hone: avg-length window
    df_out["matched_honed_original"] = df_out.apply(
        lambda r: hone_by_average_length(r["hsent"], r["gsent"], r["matched_sentence"]),
        axis=1,
    )

    # Hone: best Levenshtein substring
    # THIS WAS THE BEST MATCH.
    df_out[["matched_honed_lev", "matched_honed_lev_score"]] = df_out.apply(
        lambda r: pd.Series(hone_by_levenshtein(r["hsent"], r["matched_sentence"])),
        axis=1,
    )

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved with matches to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
