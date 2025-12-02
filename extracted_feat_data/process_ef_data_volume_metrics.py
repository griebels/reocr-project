import os
import ast
from collections import Counter

import pandas as pd

#Folder with EF 2.0 CSVs (one csv per htid, with "words_pos" column)
folder_v20 = "EF_20_data"   

#Folder with EF 2.5 CSVs (one csv per htid, with "words_pos" column)
folder_v25 = "EF_25_data"   

#Column name with the data.tokenPosCount dict
POS_TOKENS_COL = "words_pos"

#Final output
output_csv = "final_overlap_metrics_test.csv"


def extract_token_counts(pos_tokens_str):
    """
    Input (string):
        "{'the': {'DT': 49}, 'those': {'DT': 1}}"
    Output (Counter):
        Counter({'the': 49, 'those': 1})
    """
    try:
        data = ast.literal_eval(pos_tokens_str)
    except Exception:
        return Counter()

    counts = Counter()
    for token, pos_dict in data.items():
        counts[token] += sum(pos_dict.values())
    return counts


def extract_pos_counts(pos_tokens_str):
    """
    Input (string):
        "{'steps': {'NNS': 1}, 'be': {'VB': 1}, 'street': {'NN': 2}}"
    Output (Counter):
        Counter({'NN': 2, 'NNS': 1, 'VB': 1})
    """
    try:
        data = ast.literal_eval(pos_tokens_str)
    except Exception:
        return Counter()

    pos_totals = Counter()
    for token, pos_dict in data.items():
        for pos, count in pos_dict.items():
            pos_totals[pos] += count

    return pos_totals

def aggregate_book_counts_from_words_pos(csv_path, pos_tokens_col=POS_TOKENS_COL):
    """
    For a single htid csv that has a 'words_pos' column:
    For each row, parse the dict-like string and compute:
        1. word-level counts via extract_token_counts
        2. POS-level counts via extract_pos_counts
    Aggregate rows into single Counters for volume-level data
    Returns: (word_counter, pos_counter)
    """
    df = pd.read_csv(csv_path)

    if pos_tokens_col not in df.columns:
        raise ValueError(f"{pos_tokens_col} not found in {csv_path}")

    word_counter = Counter()
    pos_counter  = Counter()

    for _, row in df.iterrows():
        cell = row[pos_tokens_col]
        word_counter += extract_token_counts(cell)
        pos_counter  += extract_pos_counts(cell)

    return word_counter, pos_counter

def index_csv_folder(folder):
    """Return {filename: full_path} for all .csv files in a folder."""
    return {
        fname: os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.lower().endswith(".csv")
    }

def filename_to_htid(fname):
    """
    Convert a filename to htid.
    This works because filename is just htid with ".csv"...
    """
    return os.path.splitext(fname)[0]


#These are set for SMALL vs LARGE words and the "UNK" POS tag
SMALL_MAX_LEN = 4      #small words: <= 4
LARGE_MIN_LEN = 9      # large words: >= 9
UNK_TAG = "UNK"

files_v20 = index_csv_folder(folder_v20)
files_v25 = index_csv_folder(folder_v25)

common_files = sorted(set(files_v20) & set(files_v25))
print(f"Found {len(common_files)} books in common between EF 2.0 and EF 2.5.")

rows = []

for fname in common_files:
    path20 = files_v20[fname]
    path25 = files_v25[fname]

    # aggregate book-level counts
    words20, pos20 = aggregate_book_counts_from_words_pos(path20)
    words25, pos25 = aggregate_book_counts_from_words_pos(path25)

    ## Word metrics
    vocab20 = set(words20.keys())
    vocab25 = set(words25.keys())
    vocab_inter_words = vocab20 & vocab25

    total_tokens_20 = sum(words20.values())
    total_tokens_25 = sum(words25.values())

    # Shared tokens = intersection over counts
    shared_word_tokens = sum(min(words20[w], words25[w]) for w in vocab_inter_words)

    #DICE SIMILARITY = symmetric word-level similarity
    # calculates (2 * shared words) and divides by total words from both sets
    denom_words = total_tokens_20 + total_tokens_25
    if denom_words > 0:
        word_token_dice_similarity = 2 * shared_word_tokens / denom_words
    else:
        word_token_dice_similarity = float("nan")

    ## POS metrics
    posset20 = set(pos20.keys())
    posset25 = set(pos25.keys())
    pos_inter = posset20 & posset25

    total_pos_20 = sum(pos20.values())
    total_pos_25 = sum(pos25.values())

    shared_pos_tokens = sum(min(pos20[tag], pos25[tag]) for tag in pos_inter)

    #Getting dice similarity for pos tags
    denom_pos = total_pos_20 + total_pos_25
    if denom_pos > 0:
        pos_token_dice_similarity = 2 * shared_pos_tokens / denom_pos
    else:
        pos_token_dice_similarity = float("nan")

    unk_20 = pos20.get(UNK_TAG, 0)
    unk_25 = pos25.get(UNK_TAG, 0)

    unk_frac_20 = unk_20 / total_pos_20 if total_pos_20 else float("nan")
    unk_frac_25 = unk_25 / total_pos_25 if total_pos_25 else float("nan")

    # POS similarity after stripping UNK
    pos20_noUNK = Counter({k: v for k, v in pos20.items() if k != UNK_TAG})
    pos25_noUNK = Counter({k: v for k, v in pos25.items() if k != UNK_TAG})

    total_noUNK_20 = sum(pos20_noUNK.values())
    total_noUNK_25 = sum(pos25_noUNK.values())

    pos_tags_20 = set(pos20_noUNK)
    pos_tags_25 = set(pos25_noUNK)

    shared_noUNK = sum(
        min(pos20_noUNK[t], pos25_noUNK[t]) for t in (pos_tags_20 & pos_tags_25)
    )

    denom_noUNK = total_noUNK_20 + total_noUNK_25
    if denom_noUNK > 0:
        pos_noUNK_dice = 2 * shared_noUNK / denom_noUNK
    else:
        pos_noUNK_dice = float("nan")

    ## Large/small word metrics
    small20 = Counter()
    small25 = Counter()
    large20 = Counter()
    large25 = Counter()

    for w, c in words20.items():
        L = len(w)
        if L <= SMALL_MAX_LEN:
            small20[w] += c
        if L >= LARGE_MIN_LEN:
            large20[w] += c

    for w, c in words25.items():
        L = len(w)
        if L <= SMALL_MAX_LEN:
            small25[w] += c
        if L >= LARGE_MIN_LEN:
            large25[w] += c

    small_total_tokens_20 = sum(small20.values())
    small_total_tokens_25 = sum(small25.values())
    large_total_tokens_20 = sum(large20.values())
    large_total_tokens_25 = sum(large25.values())

    small_fraction_20 = (
        small_total_tokens_20 / total_tokens_20
        if total_tokens_20 else float("nan")
    )
    small_fraction_25 = (
        small_total_tokens_25 / total_tokens_25
        if total_tokens_25 else float("nan")
    )
    large_fraction_20 = (
        large_total_tokens_20 / total_tokens_20
        if total_tokens_20 else float("nan")
    )
    large_fraction_25 = (
        large_total_tokens_25 / total_tokens_25
        if total_tokens_25 else float("nan")
    )

    # Deltas: EF25 minus EF20
    small_word_fraction_delta = (
        small_fraction_25 - small_fraction_20
        if not (pd.isna(small_fraction_20) or pd.isna(small_fraction_25))
        else float("nan")
    )
    large_word_fraction_delta = (
        large_fraction_25 - large_fraction_20
        if not (pd.isna(large_fraction_20) or pd.isna(large_fraction_25))
        else float("nan")
    )

    htid = filename_to_htid(fname)

    rows.append(
        {
            "htid": htid,
            "filename": fname,

            # text length / expansion
            "word_total_tokens_ef20": total_tokens_20,
            "word_total_tokens_ef25": total_tokens_25,

            # global similarity
            "word_token_dice_similarity": word_token_dice_similarity,
            "pos_token_dice_similarity": pos_token_dice_similarity,
            "pos_noUNK_dice_similarity": pos_noUNK_dice,

            # failure rates
            "pos_unk_fraction_ef20": unk_frac_20,
            "pos_unk_fraction_ef25": unk_frac_25,

            # lexical shift (short / long words)
            "small_word_fraction_delta": small_word_fraction_delta,
            "large_word_fraction_delta": large_word_fraction_delta,
        }
    )

# ---------- FINAL OUTPUT ----------

final_df = pd.DataFrame(rows)
final_df.to_csv(output_csv, index=False)
print(f"Saved metrics to {output_csv}")
print(final_df.head())
