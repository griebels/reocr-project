# Much of this is adapted from Jiang et al.'s data.py
# Our updates assume you've already downloaded and used gutenbergpy.strip_headers()
# on Gutenberg texts. Our script also improves error handling, type hinting, and checks for NLTK('punkt').
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
from chapterize import Book  # assumes Book is in chapterize.py and we can import

FRONT_BACK_PREFIXES = ('CONT', 'Transcriber', 'INDEX', 'Publisher', 'FOOTNOTES', '[APPENDIX', 'THE END')

def load_text(path: Path) -> str:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        return f.read()

def get_book_from_string(text: str) -> Dict[str, Any]:
    """
    Replicates the 'get_book' function in Jiang et al.'s code 
    but starts from a txt file, not from gutenberg.acquire.load_etext.
    """
    gbk = Book(text)
    lines = gbk.lines
    last_few_lines = lines[-5:] if lines else [] # error handling

    result: Dict[str, List[List[str]]] = {'chap': [], 'footnote': []} # more explicit type hinting...
    gbk_chaps: List[List[str]] = []
    gbk_footnotes: List[List[str]] = []

    if getattr(gbk, 'headings', None) and len(gbk.headings) > 0: # error handling, in case the 'headings' doesn't even exist
        for chidx in range(len(gbk.chapters)):
            chap = gbk.chapters[chidx]            
            chap_hd = gbk.heading_str[chidx]      

            if not chap_hd.startswith(FRONT_BACK_PREFIXES):
                if chidx == len(gbk.chapters) - 1 and last_few_lines:
                    for l in last_few_lines:
                        if l not in chap:
                            chap.append(l)

                joined = ''.join(chap)
                if joined:
                    upper_count = sum(1 for c in joined if c.isupper())
                    upper_ratio = upper_count / len(joined)
                else:
                    upper_ratio = 0.0

                if upper_ratio >= 0.6:
                    continue

                gbk_chaps.append(chap)

            elif 'FOOTNOTE' in chap_hd.upper():
                if chidx == len(gbk.chapters) - 1 and last_few_lines:
                    for l in last_few_lines:
                        if l not in chap:
                            chap.append(l)
                gbk_footnotes.append(chap)
    else:
        gbk_chaps.append(text.split('\n'))

    result['chap'] = gbk_chaps
    result['footnote'] = gbk_footnotes
    return result

def get_chapter_sents(chaps: List[List[str]]) -> List[List[str]]:
    chap_sents: List[List[str]] = []
    for chap in chaps:
        chap = ['\n' if x == '' else x for x in chap]
        chap_text = ' '.join(chap)
        paragraphs = chap_text.split('\n')

        sents: List[str] = []
        for p in paragraphs:
            p = ' '.join(p.strip().split())
            if p:
                sents.extend(sent_tokenize(p))
        chap_sents.append(sents)
    return chap_sents

def get_chapter_fn(footnotes: List[List[str]]) -> List[str]:
    items: List[str] = []
    for chap in footnotes:
        chap = ['\n' if x == '' else x.strip() for x in chap]
        text = ' '.join(chap)
        for line in text.split('\n'):
            line = line.strip()
            if line:
                items.append(line)
    return items

def write_chapter_sent_file(chap_sents: List[List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as bk:
        for i, chap in enumerate(chap_sents):
            bk.write('\n'.join(chap))
            if i < len(chap_sents) - 1:
                bk.write('\n\n\n')  

def main():
    parser = argparse.ArgumentParser(description="Chapterize a folder of clean gutenberg txt files.")
    parser.add_argument('--input_dir', default='gutenbooks', help='Folder containing gutenberg .txt files.')
    parser.add_argument('--output_dir', default='chap_g_files', help='Folder to write chapterized sentence files.')
    parser.add_argument('--footnotes', action='store_true', help='Also write footnotes files if footnotes are found.')
    parser.add_argument('--nonchap_list', default='nonchap_ids.txt', help='Path to write list of files with <= 1 detected chapter.')
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    nonchap_out = Path(args.nonchap_list)

    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}")
        return

    # Doing a quick check to ensure NLTK punkt is downloaded.. if it fails, then punkt is downloaded
    try:
        sent_tokenize("Test.")
    except LookupError:
        import nltk
        nltk.download('punkt')

    nonchap: List[str] = []

    txt_files = sorted([p for p in in_dir.glob('*.txt') if p.is_file()])
    if not txt_files:
        print(f"No .txt files found in {in_dir}")
        return

    for idx, fpath in enumerate(txt_files, start=1):
        try:
            text = load_text(fpath)
            result = get_book_from_string(text)

            if len(result['chap']) > 1:
                chap_sents = get_chapter_sents(result['chap'])
                out_file = out_dir / f"{fpath.stem}.txt"
                write_chapter_sent_file(chap_sents, out_file)

                if args.footnotes and len(result['footnote']) > 0:
                    fn_items = [get_chapter_fn(result['footnote'])]  # one "chapter" of footnotes
                    fn_out_file = out_dir / f"{fpath.stem}_footnote.txt"
                    write_chapter_sent_file(fn_items, fn_out_file)
            else:
                # anything with <= 1 chapters...
                nonchap.append(fpath.name)

            print(f"[{idx}/{len(txt_files)}] Done\t{fpath.name}")

        except Exception as e:
            print(f"ERROR processing {fpath.name}: {e}")
            nonchap.append(f"{fpath.name}\tERROR:{e}")

    if nonchap:
        nonchap_out.parent.mkdir(parents=True, exist_ok=True)
        with nonchap_out.open('w', encoding='utf-8') as f:
            f.write('\n'.join(nonchap))
        print(f"Here is the non-chapterized book list: {nonchap_out}")

if __name__ == '__main__':
    main()
