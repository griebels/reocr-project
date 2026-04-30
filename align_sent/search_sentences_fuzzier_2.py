#!/usr/bin/env python3
"""
search_sentences_fuzzier.py

Search a directory of HathiTrust ZIP volumes for ~N target sentences
(typically ~170k) that were originally extracted from clean Project Gutenberg
texts, USING THE GUTENBERG SENTENCE AS THE SEARCH OBJECT, NOT THE HATHITRUST SENTENCE.

TO SEARCH USING THE FIRST FOUND HT SENTENCE, USE VERSION 1 (NO NUMBER) OF THE IDENTICALLY NAMED SCRIPT

The goal is to find the corresponding OCR'd sentences in HTDL to study
OCR quality and its effect on downstream tasks.

For each volume:
  - Only match sentences whose 'hid' matches the volume's HTID (zip filename).
  - Use spaCy to split page text into sentences and tokenize.
  - Use a normalized token representation for matching (lowercased, quote-normalized,
    whitespace stripped) while preserving raw OCR text in output.
  - First try exact token-sequence matches via hashing.
  - If that fails, use fuzzy token-level Levenshtein distance with:
      * dynamic length windows (larger for longer sentences),
      * dynamic max distance (larger for longer sentences),
      * relaxed first/last-token checks for long sentences.

Outputs:
  - matches CSV: unambiguous matches (one HT sentence per PG target)
  - ambiguous CSV: ambiguous matches (multiple candidates)
  - log CSV: zip/page errors and "sentence_not_found" entries
"""

import os
import zipfile
import hashlib
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import spacy
from tqdm import tqdm


# ------------------------------------------------------------------------------
# GLOBAL SPACY OBJECT (LAZY INIT PER PROCESS)
# ------------------------------------------------------------------------------

_nlp = None


def get_nlp():
    """
    Lazily create and cache a minimal spaCy English pipeline
    with only a sentencizer for sentence segmentation.
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ------------------------------------------------------------------------------
# TOKENIZATION HELPERS
# ------------------------------------------------------------------------------

def tokenize(text):
    """
    Tokenize a piece of text into a list of tokens, preserving punctuation.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [t.text for t in doc]


def tokenize_sents(text):
    """
    Split text into sentences using spaCy, then tokenize each sentence.

    Now includes contraction-normalization so that token sequences
    are more compatible with Gutenberg's formatting.
    """
    nlp = get_nlp()
    doc = nlp(text)

    for sent in doc.sents:
        tokens_raw = [t.text for t in sent]

        # NEW: normalize contractions BEFORE normalization/hashing
        tokens_raw = normalize_contractions(tokens_raw)

        yield tokens_raw



# ------------------------------------------------------------------------------
# NORMALIZATION, HASHING, AND DISTANCE
# ------------------------------------------------------------------------------

def normalize_token(t):
    if t is None:
        return None

    s = t.strip().lower()

    # Normalize quotes
    quote_map = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "`": "'",
    }
    if s in quote_map:
        s = quote_map[s]

    # Normalize dashes
    dash_map = {
        "—": "-", "–": "-", "−": "-", "‒": "-",
    }
    if s in dash_map:
        s = dash_map[s]

    # Normalize repeated punctuation => single
    # e.g. "!!" → "!", ".." → "."
    while len(s) > 1 and all(ch == s[0] for ch in s):
        s = s[0]

    # Normalize spacing around punctuation, make commas uniform
    if s in {",", ",", " ,"}:
        return ","

    # Merge tokens like ".?" → "?". OCR often outputs artifacts like ".?"
    if s in {".?", "?.", ".! ", "!.", "?!", "!?"}:
        return s[-1]

    # Normalize hyphen variants
    if s == "‐":  # Unicode weird hyphen
        return "-"

    return s

def normalize_contractions(tokens):
    """
    Merge common spaCy tokenizations of contractions so that token sequences
    match Gutenberg text more reliably.

    Examples:
      ["do", "n't"] → ["don't"]
      ["can", "not"] → ["cannot"]
      ["I", "'m"] → ["I'm"]
      ["they", "'re"] → ["they're"]
    """
    out = []
    skip = False

    for i in range(len(tokens)):
        if skip:
            skip = False
            continue

        tok = tokens[i]

        # don't
        if tok == "do" and i + 1 < len(tokens) and tokens[i + 1] == "n't":
            out.append("don't")
            skip = True
            continue

        # can't
        if tok == "ca" and i + 1 < len(tokens) and tokens[i + 1] == "n't":
            out.append("can't")
            skip = True
            continue

        # cannot (often useful in OCR)
        if tok == "can" and i + 1 < len(tokens) and tokens[i + 1] == "not":
            out.append("cannot")
            skip = True
            continue

        # I'm
        if tok == "i" and i + 1 < len(tokens) and tokens[i + 1] == "'m":
            out.append("i'm")
            skip = True
            continue

        # you're
        if tok == "you" and i + 1 < len(tokens) and tokens[i + 1] == "'re":
            out.append("you're")
            skip = True
            continue

        # we're
        if tok == "we" and i + 1 < len(tokens) and tokens[i + 1] == "'re":
            out.append("we're")
            skip = True
            continue

        # they're
        if tok == "they" and i + 1 < len(tokens) and tokens[i + 1] == "'re":
            out.append("they're")
            skip = True
            continue

        # I'll
        if tok == "i" and i + 1 < len(tokens) and tokens[i + 1] == "'ll":
            out.append("i'll")
            skip = True
            continue

        out.append(tok)

    return out



def tokens_hash(tokens):
    """
    Compute a stable MD5 hash for a sequence of tokens. The tokens
    are assumed to already be normalized for matching.
    """
    joined = "\u241F".join(tokens)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def token_levenshtein(a, b, maxdist):
    """
    Compute token-level Levenshtein distance between lists a and b,
    with early stopping when distance > maxdist.
    """
    len_a, len_b = len(a), len(b)

    # Quick reject by length difference
    if abs(len_a - len_b) > maxdist:
        return maxdist + 1

    prev_row = list(range(len_b + 1))

    for i, tok_a in enumerate(a, start=1):
        curr_row = [i]
        row_min = curr_row[0]

        for j, tok_b in enumerate(b, start=1):
            cost = 0 if tok_a == tok_b else 1
            insertion = curr_row[j - 1] + 1
            deletion = prev_row[j] + 1
            substitution = prev_row[j - 1] + cost
            val = min(insertion, deletion, substitution)
            curr_row.append(val)

            if val < row_min:
                row_min = val

        if row_min > maxdist:
            return maxdist + 1

        prev_row = curr_row

    return prev_row[-1]


def diff_tokens(original_tokens, page_tokens):
    """
    Produce a simple token-level diff string showing which tokens were
    removed (-token) or added (+token), useful for inspecting differences
    between Gutenberg and OCR sentences.
    """
    from itertools import zip_longest

    diffs = []
    for o, p in zip_longest(original_tokens, page_tokens, fillvalue=None):
        if o == p:
            continue
        if o is not None:
            diffs.append(f"-{o}")
        if p is not None:
            diffs.append(f"+{p}")
    return " ".join(diffs)


# ------------------------------------------------------------------------------
# BUILD PER-VOLUME SENTENCE INDEX
# ------------------------------------------------------------------------------

def build_sentence_index(sent_path):
    """
    Read the CSV of target sentences (test_sents) and build a per-volume index.

    Each row of the CSV is expected to have:
      - hsent : the HTDL aligned sentence
      - gsent : the Project Gutenberg aligned sentence
      - gid   : Gutenberg item id
      - hid   : the HTDL volume id (HTID) for that sentence
      - csv_source : the source CSV of the aligned target sentence
    We also preserve the original row index to report in outputs.

    Returns:
      - volume_index: dict[hid] -> {"meta_list": [...], "by_hash": {...}}
      - df: the full DataFrame with an 'index' column indicating row number
    """
    df = pd.read_csv(sent_path, dtype=str)
    df = df.reset_index()  # keep original row index in df["index"]

    volume_index = {}

    for _, row in df.iterrows():
        hid = row.get("hid")
        gsent = row.get("gsent")
        hsent = row.get("hsent")
        gid = row.get("gid")
        row_index = int(row["index"])
        csv_source = row.get("csv_source")

        # Skip rows with missing or empty sentences or volume ids
        if not isinstance(hid, str) or not hid.strip():
            continue
        if not isinstance(gsent, str) or not gsent.strip():
            continue

        tokens_raw = tokenize(gsent)
        tokens_norm = [normalize_token(t) for t in tokens_raw if t.strip() != ""]
        thash = tokens_hash(tokens_norm)

        meta = {
            "row_index": row_index,
            "csv_source": csv_source,
            "gid": gid,
            "htid": hid,
            "gsent": gsent,
            "hsent": hsent,
            "tokens_raw": tokens_raw,
            "tokens_norm": tokens_norm,
            "length": len(tokens_norm),
            "first": tokens_norm[0] if tokens_norm else "",
            "last": tokens_norm[-1] if tokens_norm else "",
            "hash": thash,
        }

        if hid not in volume_index:
            volume_index[hid] = {"meta_list": [], "by_hash": {}}

        volume_index[hid]["meta_list"].append(meta)
        volume_index[hid]["by_hash"].setdefault(thash, []).append(meta)

    return volume_index, df


# ------------------------------------------------------------------------------
# MATCH A SINGLE PAGE SENTENCE IN A VOLUME
# ------------------------------------------------------------------------------

def match_sentence_in_volume(page_tokens_raw, vol_index_for_htid, maxdist, lenwindow):
    """
    Attempt to match a single OCR page-sentence to the Gutenberg target sentences.
    Uses raw OCR tokens for output, normalized tokens for matching.
    """

    if not vol_index_for_htid:
        return []

    # NEW: normalize contractions BEFORE building normalized tokens
    page_tokens_raw = normalize_contractions(page_tokens_raw)

    # Normalize and strip whitespace-only tokens
    page_tokens_norm = [normalize_token(t) for t in page_tokens_raw if t.strip() != ""]
    if not page_tokens_norm:
        return []

    meta_list = vol_index_for_htid["meta_list"]
    by_hash = vol_index_for_htid["by_hash"]

    candidates = []

    # ---- EXACT MATCH: hash match on normalized tokens ----
    phash = tokens_hash(page_tokens_norm)
    if phash in by_hash:
        for meta in by_hash[phash]:
            candidates.append({"meta": meta, "distance": 0})

    # ---- FUZZY MATCH (only if exact match failed) ----
    if not candidates and maxdist > 0:

        len_page = len(page_tokens_norm)
        first_page = page_tokens_norm[0]
        last_page = page_tokens_norm[-1]
        QUOTE_TOKENS = {'"', "'"}

        for meta in meta_list:
            len_meta = meta["length"]

            # (A) Dynamic length window
            base_window = lenwindow
            dyn_window = int(0.40 * min(len_page, len_meta))   # 40% now
            max_lenwindow = max(base_window, dyn_window)

            if abs(len_page - len_meta) > max_lenwindow:
                continue

            # (B) Relaxed first/last checks for long-ish sentences
            first_match = (
                first_page == meta["first"]
                or {first_page, meta["first"]} & QUOTE_TOKENS
            )
            last_match = (
                last_page == meta["last"]
                or {last_page, meta["last"]} & QUOTE_TOKENS
            )

            if not (first_match or last_match):
                if len_meta <= 25:   # WAS 40 — now more permissive
                    continue

            # (C) Dynamic edit distance threshold
            dyn_maxdist = max(maxdist, int(0.20 * min(len_page, len_meta)))
            dyn_maxdist = min(dyn_maxdist, 12)  # WAS 8

            # (D) Token-level normalized Levenshtein distance
            dist = token_levenshtein(page_tokens_norm, meta["tokens_norm"], dyn_maxdist)
            if dist <= dyn_maxdist:
                candidates.append({"meta": meta, "distance": dist})

    return candidates

# ------------------------------------------------------------------------------
# PROCESS A SINGLE ZIP VOLUME
# ------------------------------------------------------------------------------

def process_zip(args):
    """
    Process a single ZIP volume:
      - derive htid from zip filename (strip '.zip'),
      - skip if no target sentences for this htid,
      - read each .txt file, split into sentence token lists,
      - attempt to match each page sentence via exact + fuzzy matching.

    Returns:
      matches:   list of unambiguous matches
      ambiguous: list of ambiguous matches
      logs:      list of log entries (errors + ambiguous info)
    """
    zip_path, volume_index, maxdist, lenwindow = args

    matches = []
    ambiguous = []
    logs = []

    zname = os.path.basename(zip_path)
    htid = zname.rsplit(".zip", 1)[0]

    # If no sentences for this volume, skip entirely
    if htid not in volume_index:
        return matches, ambiguous, logs

    vol_index_for_htid = volume_index[htid]

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for fn in z.namelist():
                if not fn.lower().endswith(".txt"):
                    continue

                try:
                    raw = z.read(fn).decode("utf-8", errors="replace")

                    for page_tokens_raw in tokenize_sents(raw):
                        if not page_tokens_raw:
                            continue

                        cands = match_sentence_in_volume(
                            page_tokens_raw, vol_index_for_htid, maxdist, lenwindow
                        )
                        if not cands:
                            continue

                        found_sentence = " ".join(page_tokens_raw)

                        # Unambiguous: exactly one candidate
                        if len(cands) == 1:
                            meta = cands[0]["meta"]
                            matches.append({
                                "htid": htid,
                                "csv_source": meta["csv_source"],
                                "found_hsent": found_sentence,
                                "gid": meta["gid"],
                                "source_row_index": meta["row_index"],
                                "target_gsent": meta["gsent"],
                            })

                        # Ambiguous: more than one candidate
                        else:
                            for c in cands:
                                meta = c["meta"]
                                dist = c["distance"]
                                diff = diff_tokens(meta["tokens_raw"], page_tokens_raw)

                                ambiguous.append({
                                    "htid": htid,
                                    "found_hsent": found_sentence,
                                    "gid": meta["gid"],
                                     "csv_source": meta["csv_source"],
                                    "source_row_index": meta["row_index"],
                                    "target_gsent": meta["gsent"],
                                    "distance": dist,
                                    "diff": diff,
                                })

                                logs.append({
                                    "log_type": "ambiguous_match",
                                    "zip": zname,
                                    "page": fn,
                                    "htid": htid,
                                    "csv_source": meta["csv_source"],
                                    "target_hsent": meta["hsent"],
                                    "target_gsent": meta["gsent"],
                                    "gid": meta["gid"],
                                    "row_index": meta["row_index"],
                                    "error": f"{len(cands)} candidates (distance={dist})",
                                    "found_hsent": found_sentence,
                                    "distance": dist,
                                    "diff": diff,
                                })

                except Exception as e:
                    logs.append({
                        "log_type": "page_error",
                        "zip": zname,
                        "page": fn,
                        "htid": htid,
                        "target_hsent": "",
                        "csv_source": meta["csv_source"],
                        "gid": "",
                        "row_index": "",
                        "error": str(e),
                        "found_hsent": "",
                        "distance": "",
                        "diff": "",
                    })

    except Exception as e:
        logs.append({
            "log_type": "zip_error",
            "zip": zname,
            "page": "",
            "htid": htid,
            "target_hsent": "",
             "csv_source": meta["csv_source"],
            "gid": "",
            "row_index": "",
            "error": str(e),
            "found_hsent": "",
            "distance": "",
            "diff": "",
        })

    return matches, ambiguous, logs


# ------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------------------------

def main():
    """
    Command-line entry point:
      - parse arguments,
      - build sentence index,
      - scan all ZIP volumes in parallel,
      - write matches, ambiguous matches, and logs to CSV.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fuzzy search for Gutenberg sentences in HTDL OCR page text."
    )
    parser.add_argument("--sentfile", required=True,
                        help="CSV of target sentences with at least columns: gsent,gid,hid")
    parser.add_argument("--zipdir", required=True,
                        help="Directory containing HTDL volume .zip files")
    parser.add_argument("--output", required=True,
                        help="Output CSV for unambiguous matches")
    parser.add_argument("--ambigfile", required=True,
                        help="Output CSV for ambiguous matches")
    parser.add_argument("--logfile", required=True,
                        help="Output CSV for logs (errors + sentence_not_found)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker processes (default: 8)")
    parser.add_argument("--maxdist", type=int, default=2,
                        help="Base maximum token edit distance (default: 2)")
    parser.add_argument("--lenwindow", type=int, default=5,
                        help="Base length difference window (default: 5)")

    args = parser.parse_args()

    # Build the per-volume sentence index from the sentfile.
    print("Building per-volume sentence index from target sentences...")
    volume_index, df = build_sentence_index(args.sentfile)
    print(f"Indexed sentences for {len(volume_index)} distinct hids.\n")

    # Collect all .zip paths in the given directory.
    zip_paths = sorted(str(p) for p in Path(args.zipdir).glob("*.zip"))
    print(f"Found {len(zip_paths)} ZIP volumes to scan.\n")

    tasks = [(zp, volume_index, args.maxdist, args.lenwindow) for zp in zip_paths]

    all_matches = []
    all_ambig = []
    all_logs = []

    # Process all zip volumes in parallel with a progress bar.
    with Pool(processes=args.workers) as pool:
        for matches, ambiguous, logs in tqdm(
            pool.imap_unordered(process_zip, tasks),
            total=len(tasks),
            desc="Searching volumes",
            unit="vol"
        ):
            all_matches.extend(matches)
            all_ambig.extend(ambiguous)
            all_logs.extend(logs)

    # Write unambiguous matches
    matches_df = pd.DataFrame(all_matches)
    matches_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(matches_df)} unambiguous matches to {args.output}")

    # Write ambiguous matches
    ambig_df = pd.DataFrame(all_ambig)
    ambig_df.to_csv(args.ambigfile, index=False)
    print(f"Wrote {len(ambig_df)} ambiguous matches to {args.ambigfile}")

    # Determine which source rows were matched at all (including ambiguous)
    matched_rows = set()
    if not matches_df.empty:
        matched_rows.update(matches_df["source_row_index"].astype(int).tolist())
    if not ambig_df.empty:
        matched_rows.update(ambig_df["source_row_index"].astype(int).tolist())

    df = df.copy()
    df["index"] = df["index"].astype(int)
    all_indices = set(df["index"].tolist())
    missing_indices = all_indices - matched_rows

    # Log sentences that were not matched in any way
    for idx in sorted(missing_indices):
        row = df.loc[df["index"] == idx].iloc[0]
        all_logs.append({
            "log_type": "sentence_not_found",
            "zip": "",
            "page": "",
            "htid": row.get("hid", ""),
            "target_hsent": row.get("hsent", ""),
            "target_gsent": row.get("gsent", ""),
            "gid": row.get("gid", ""),
            "row_index": int(row["index"]),
            "error": "No match found (exact or fuzzy) in corresponding volume",
            "found_hsent": "",
            "distance": "",
            "diff": "",
        })

    # Write unified log
    log_df = pd.DataFrame(all_logs)
    log_df.to_csv(args.logfile, index=False)
    print(f"Wrote {len(log_df)} log entries to {args.logfile}")


if __name__ == "__main__":
    main()
