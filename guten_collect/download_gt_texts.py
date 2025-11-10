#!/usr/bin/env python3
"""
Downloads and cleans Gutenberg txt files using gutenbergpy.
Takes a txt file with gutenberg IDs (one per line) and saves
cleaned txt files to a folder as "id.txt".
If gutenbergpy fails, falls back to direct HTTP download.

Usage:
    python download_gt_texts.py --id_file gutenberg_ids.txt --output_dir gutenberg_txts
"""
# Make sure your environment contains: gutenbergpy
# Install if needed: pip install gutenbergpy
import argparse
from pathlib import Path
import time
import requests
import gutenbergpy.textget as tg


def fetch_pg_bytes(gid, timeout=20):
    """
    Tries a few known URL patterns for getting Gutenberg texts.
    Had to resort to this when gutenbergpy failed for one text and the
    there was a "Raise None" error.
    """
    gid = str(gid).strip()
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt.utf8",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf8",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}-0.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}-0.txt.utf8",
    ]

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content and len(r.content) > 200:
                return r.content
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Couldn't get PG text for id {gid}. Due to error: {last_err}")


def main(id_file, output_dir="/gutenberg_txt"):
    """
    fetches and saves each Gutenberg text from the list of IDs.
    """
    id_path = Path(id_file)
    if not id_path.exists():
        raise FileNotFoundError(f"couldn't find id file: {id_file}")

    with open(id_file, "r", encoding="utf-8") as f:
        gut_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not gut_ids:
        raise ValueError("No valid gutenberg ids found in file.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    failures = []

    print(f"Starting download of {len(gut_ids)} Gutenberg texts...")
    for gid in gut_ids:
        try:
            try:
                raw = tg.get_text_by_id(gid)
            except TypeError:
                # Work around the 'raise None' by fetching directly
                raw = fetch_pg_bytes(gid)

            clean = tg.strip_headers(raw).decode("utf-8", errors="replace").strip()
            (out / f"{gid}.txt").write_text(clean, encoding="utf-8")
            print(f"Saved {gid}.txt")  # this might be too verbose, but leaving it in for now.
            time.sleep(0.5)  # be polite

        except Exception as e:
            failures.append((gid, str(e)))
            print(f"Failed on {gid}: {e}")

    print(f"\nDone. Saved {len(gut_ids)-len(failures)} / {len(gut_ids)} files.")
    if failures:
        print("Failed IDs (first 10):")
        for g, err in failures[:10]:
            print(f"- {g}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and clean PG txt files.")
    parser.add_argument("--id_file", required=True, help="txt file path containing Gutenberg IDs (one per line).")
    parser.add_argument("--output_dir", default="gutenberg_txts", help="output folder")
    args = parser.parse_args()

    main(args.id_file, args.output_dir)
