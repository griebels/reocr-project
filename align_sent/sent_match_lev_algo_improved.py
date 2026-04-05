#!/usr/bin/env python3
"""
Match HathiTrust OCR sentences to target sentences from a csv.

The script normalizes text, filters likely candidates, scores possible matches,
and returns the best sentence-level or short window-level match for each target
sentence. 
Then does a final character-level refinement step and writes the
results to a csv.
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
from rapidfuzz import fuzz


# Data imports here.
TXT_DIR = Path("346_highlowpairs_hathi_vols_concat")
CSV_PATH = "ht_pg_sents_df_167079_2.csv"
OUTPUT_PATH = "found_in_january_run_improved_ALL.csv"


def normalize_text(s: str) -> str:
    """
    Some fast and loose normalization for quick matching:
    - Unicode normalize
    - lowercase everything
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


# load the text files, split on sentences
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


def get_ngrams(text: str, n: int = 2) -> set:
    """
    Return the set of word n-grams in a text string.
    """
    words = text.split()
    if len(words) < n:
        return set(words)
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def score_candidate_improved(query_norm: str, cand_norm: str) -> dict:
    """
    Scores a candidate match with a small set of overlap and similarity measures,
    then combine them into a single score.
    - containment: word overlap (same as before, percentage where containment = |query_words ∩ cand_words| / |query_words|)
    - bigram_overlap: bigram overlap (preserves some word order)
    - lev_similarity: normalized Levenshtein similarity
    - combined: weighted combination = 0.4*containment, 0.3*bigram, 0.3*lev_sim
    """
    if not query_norm or not cand_norm:
        return {
            "containment": 0.0,
            "bigram_overlap": 0.0,
            "lev_similarity": 0.0,
            "combined": 0.0
        }

    q_words = query_norm.split()
    c_words = cand_norm.split()
    
    if not q_words or not c_words:
        return {
            "containment": 0.0,
            "bigram_overlap": 0.0,
            "lev_similarity": 0.0,
            "combined": 0.0
        }

    # Word containment
    q_set = set(q_words)
    c_set = set(c_words)
    containment = len(q_set & c_set) / len(q_set) if q_set else 0.0

    # Bigram overlap (helps prove taht word order was preserved)
    q_bigrams = get_ngrams(query_norm, n=2)
    c_bigrams = get_ngrams(cand_norm, n=2)
    bigram_overlap = len(q_bigrams & c_bigrams) / len(q_bigrams) if q_bigrams else 0.0

    # Partial Ratio Levenshtein to answer: How similar is the best-matching substring of matched_sent to hsent?
    # Use partial_ratio which finds best matching substring
    lev_similarity = fuzz.partial_ratio(query_norm, cand_norm) / 100.0

    # Combined score: weighted average
    # Higher weight on containment and bigrams for initial filtering (could be changed)
    combined = (
        0.4 * containment +
        0.3 * bigram_overlap +
        0.3 * lev_similarity
    ) * 100.0

    return {
        "containment": containment,
        "bigram_overlap": bigram_overlap,
        "lev_similarity": lev_similarity,
        "combined": combined
    }


def anchor_filter_indices_improved(
    hsent_norm: str, 
    cand_norm_list: list[str], 
    min_len: int = 4,
    use_multiple_anchors: bool = True
) -> list[int]:
    """
    Filter candidate sentences by requiring at least one anchor word from the query.
    - uses 4-character min length (significant word)
    - multiple possible anchor words (first, middle, last significant words)
    - Require at least one anchor to be present

    """
    words = hsent_norm.split()
    if not words:
        return list(range(len(cand_norm_list)))

    # Select anchor words (filter out very short words)
    significant_words = [w for w in words if len(w) >= min_len]
    
    if not significant_words:
        # Fall back to middle word
        anchor_words = [words[len(words) // 2]]
    elif len(significant_words) == 1:
        anchor_words = [significant_words[0]]
    elif len(significant_words) == 2:
        anchor_words = [significant_words[0], significant_words[-1]]
    else:
        # Use first, middle, and last significant words
        anchor_words = [
            significant_words[0],
            significant_words[len(significant_words) // 2],
            significant_words[-1]
        ]

    # Find candidates containing at least one anchor
    anchor_indices = []
    for i, cand in enumerate(cand_norm_list):
        cand_words = set(cand.split())
        # Check if any anchor appears
        if any(anchor in cand_words for anchor in anchor_words):
            anchor_indices.append(i)

    # return the indices of those sents with anchors, else return all candidates
    return anchor_indices if anchor_indices else list(range(len(cand_norm_list)))


# produces matched_honed_lev
def hone_by_levenshtein(
    hsent: str,
    matched_sentence: str,
    max_window_expand: int = 50,
    min_similarity: float = 0.20,
):
    """
    Find the best-matching substring of "matched_sentence" relative to "hsent"
    using RapidFuzz Levenshtein normalized similarity.

    Returns: (span, similarity) where similarity is between 0-1.

    Window expansion is a guestimate about how few or how many characters more/less than hsent might be needed
    to assess whether it matches. With bad OCR, the original hsent or the matched_sent might
    be slightly corrupted, so they won't align completely. So window expansion can be adjusted for
    more flexibility.

    Similarly, min_similarity can be adjusted up or down if it's taking in too many
    non-matching options or we're not getting a lot of matched_honed_lev outputs.
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


# Worker function that runs for each process
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
            matches.append({
                "row_index": row_idx,
                "matched_sentence": None,
                "matched_sent_idx": None,
                "score_containment": 0.0,  # initialzie
                "score_bigram": 0.0,   # initialize
                "score_lev": 0.0,  # initialize
                "score_combined": 0.0,  #initialize
            })
        return matches

    # thresholds that can be tuned here
    MIN_CONTAINMENT = 0.10      # at least 10 percent word overlap
    MIN_COMBINED = 12.0         # minimum combined score (weighted score)
    MAX_WINDOW_SIZE = 20         # searches 3 sentences on either side of candidate
    
    n_sent = len(sentences_orig)

    for row_idx, row in group.iterrows():
        hsent_norm = row["hsent_norm"]

        if not sentences_norm:
            matches.append({
                "row_index": row_idx,
                "matched_sentence": None,
                "matched_sent_idx": None,
                "score_containment": 0.0,
                "score_bigram": 0.0,
                "score_lev": 0.0,
                "score_combined": 0.0,
            })
            continue

        # FIRST PASS: filtering for candidates with the anchor words
        candidate_indices = anchor_filter_indices_improved(hsent_norm, sentences_norm)

        # SECOND PASS: best single sentence: Find best single sentence by looking at containment, bigrams, lev, and combined
        # first initialize the scores, then loop through each candidate
        best_idx = None
        best_combined = -1.0
        best_scores = None

        for i in candidate_indices:
            scores = score_candidate_improved(hsent_norm, sentences_norm[i])

            # Early rejection based on containment
            if scores["containment"] < MIN_CONTAINMENT:
                continue

            if scores["combined"] > best_combined:
                best_idx = i
                best_combined = scores["combined"]
                best_scores = scores

        # if none of the candidates pass the tests, then there is no matched sent:
        if best_idx is None or best_combined < MIN_COMBINED:
            matches.append({
                "row_index": row_idx,
                "matched_sentence": None,
                "matched_sent_idx": None,
                "score_containment": 0.0,
                "score_bigram": 0.0,
                "score_lev": 0.0,
                "score_combined": 0.0,
            })
            continue

        # THIRD PASS: Expand window around best match
        center = best_idx
        
        windows = []
        # Try different window sizes, and more flexible windows
        for window_size in range(MAX_WINDOW_SIZE + 1):
            for offset in range(-window_size, 1):
                start_idx = center + offset
                end_idx = start_idx + window_size
                
                if start_idx < 0 or end_idx >= n_sent:
                    continue
                    
                idx_list = list(range(start_idx, end_idx + 1))
                if idx_list not in windows:
                    windows.append(idx_list)

        best_window_indices = None
        best_window_score = -1.0
        best_window_scores = None

        for idx_list in windows:
            window_norm = " ".join(sentences_norm[i] for i in idx_list)
            scores = score_candidate_improved(hsent_norm, window_norm)
            
            if scores["combined"] > best_window_score:
                best_window_score = scores["combined"]
                best_window_indices = idx_list
                best_window_scores = scores

        if best_window_indices is None or best_window_score < MIN_COMBINED:
            matches.append({
                "row_index": row_idx,
                "matched_sentence": None,
                "matched_sent_idx": None,
                "score_containment": 0.0,
                "score_bigram": 0.0,
                "score_lev": 0.0,
                "score_combined": 0.0,
            })
            continue

        matched_sentence = " ".join(sentences_orig[i] for i in best_window_indices)

        matches.append({
            "row_index": row_idx,
            "matched_sentence": matched_sentence,
            "matched_sent_idx": center,
            "score_containment": best_window_scores["containment"],
            "score_bigram": best_window_scores["bigram_overlap"],
            "score_lev": best_window_scores["lev_similarity"],
            "score_combined": best_window_scores["combined"],
        })

    return matches


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
        for i, fut in enumerate(as_completed(futures)):
            all_matches.extend(fut.result())
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(futures)} volumes...")

    matches_df = pd.DataFrame(all_matches).set_index("row_index")

    # Join match results back to df
    df_out = df.join(
        matches_df[[
            "matched_sentence", "matched_sent_idx",
            "score_containment", "score_bigram", "score_lev", "score_combined"
        ]],
        how="left",
    )

    # Hone: best Levenshtein substring
    df_out[["matched_honed_lev", "matched_honed_lev_score"]] = df_out.apply(
        lambda r: pd.Series(hone_by_levenshtein(r["hsent"], r["matched_sentence"]))
        if pd.notna(r["matched_sentence"]) else pd.Series([None, None]),
        axis=1,
    )

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved with matches to: {OUTPUT_PATH}")
    
    # Print some statistics
    matched = df_out["matched_sentence"].notna().sum()
    total = len(df_out)
    print(f"Matched: {matched}/{total} ({100*matched/total:.1f}%)")
    
    if matched > 0:
        print(f"\nScore statistics (for matched rows):")
        print(df_out[df_out["matched_sentence"].notna()][
            ["score_containment", "score_bigram", "score_lev", "score_combined"]
        ].describe())


if __name__ == "__main__":
    main()