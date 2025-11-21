# -----------------------------------------------------------
# !/usr/bin/env python2.7
#  -*- coding: utf-8 -*-
# This is a revised version of xer (https://github.com/jpuigcerver/xer).
# In the original version of xer, the input files should be aligned at 
# sentence level (each line in a file is a sentence).
# In this revised version, we removed this constraint and do not output 
# sentence-level error rate.
# ---------------------------------------------------------------

### The seed of this script comes from Gutenberg_HTRC_Code and is used to test replicability of the original study.


from jiwer import wer as jiwer_wer
import Levenshtein
import editdistance
import evaluate

import argparse
import logging
import re, unicodedata



logging.basicConfig(
    format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

def levenshtein(u, v):
    prev = None
    
    #curr = [0] + range(1, len(v) + 1)        ##python 2.7
    curr = [idx for idx in range(len(v)+1)]   ##python 3.7
    
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
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
    return curr[len(v)], curr_ops[len(v)]

def load_file(fname):
    try:
        f = open(fname, 'r')
        data = []
        for line in f:
            data.append(line.rstrip('\n').rstrip('\r'))
        f.close()
    except:
        logging.error('Error reading file "%s"', fname)
        exit(1)
    return ' '.join(data)

def safe_div(num, den):
    return float('nan') if den == 0 else num / den

def test_other_metrics(ref: str, hyp: str):
    """
    Returns a dict with CER/WER from several libraries:
      - cer_editdistance: editdistance.eval / len(ref)
      - cer_levenshtein: Levenshtein.distance / len(ref)
      - wer_jiwer: jiwer.wer(ref, hyp)
      - wer_evaluate: HF evaluate 'wer'
      - cer_evaluate: HF evaluate 'cer'
    """
    # Load once if you like; or cache globally
    hf_wer = evaluate.load("wer")
    hf_cer = evaluate.load("cer")

    cer_edit = safe_div(editdistance.eval(ref, hyp), len(ref))
    cer_lev  = safe_div(Levenshtein.distance(ref, hyp), len(ref))

    wer_j = jiwer_wer(ref, hyp)

    # HuggingFace evaluate expects lists
    wer_hf = hf_wer.compute(predictions=[hyp], references=[ref])
    cer_hf = hf_cer.compute(predictions=[hyp], references=[ref])

    return {
        "cer_editdistance": cer_edit,
        "cer_levenshtein": cer_lev,
        "wer_jiwer": wer_j,
        "wer_evaluate": wer_hf,
        "cer_evaluate": cer_hf,
    }


# cleaning code:

def normalize_text(text):
    # fix unicode (quotes, dashes, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # remove weird OCR chars
    text = re.sub(r'[@^~•·§¤]', '', text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    # remove space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # remove linebreak hypphens and join words
    text = re.sub(r'-\s*\n\s*', '', text)

    # lowercase for fair comparison
    text = text.lower()

    return text.strip()







def main():
    parser = argparse.ArgumentParser(description='Compute useful evaluation metrics (CER, WER, SER, ...)')
    parser.add_argument('-g', '--guten_path', type=str, default=None, help='gutenberg file path')
    parser.add_argument('-ht', '--htrc_path', type=str, default=None, help='htrc file path')
    args = parser.parse_args()
    
    gtxt = load_file(args.guten_path)
    htxt = load_file(args.htrc_path)

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0

    htxt = normalize_text(htxt)
    gtxt = normalize_text(gtxt)

    print("htxt", htxt)
    print("gtxt", gtxt)

    
    # update CER statistics
    _, (s, i, d) = levenshtein(gtxt, htxt)
    cer_s += s
    cer_i += i
    cer_d += d
    cer_n += len(gtxt)

    # update WER statistics
    _, (s, i, d) = levenshtein(gtxt.split(), htxt.split())
    wer_s += s
    wer_i += i
    wer_d += d
    wer_n += len(gtxt.split())

    others = test_other_metrics(gtxt, htxt)
    
    if cer_n > 0:
        print ('CER: %g%%, WER: %g%%' % (
            (100.0 * (cer_s + cer_i + cer_d)) / cer_n,
            (100.0 * (wer_s + wer_i + wer_d)) / wer_n
            ))
        

        print("=== Third-party ===")
        print(f"CER (editdistance): {100*others['cer_editdistance']:.3f}%")
        print(f"CER (python-Levenshtein): {100*others['cer_levenshtein']:.3f}%")
        print(f"CER (HF evaluate): {100*others['cer_evaluate']:.3f}%")
        print(f"WER (jiwer): {100*others['wer_jiwer']:.3f}%")
        print(f"WER (HF evaluate): {100*others['wer_evaluate']:.3f}%")



if __name__ == '__main__':
    main()