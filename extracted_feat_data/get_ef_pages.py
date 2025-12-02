import os
import csv
import bz2
import json
import subprocess
import requests
import argparse

"""
This script takes a txt file with HTIDs, one HTID per line.
It uses either the EF-API or RSYNC method (designated by user) 
to get page-level HT Extracted Features to output into a CSV.

**Note** - for the rsync method, must be using Linux environment with
rsync installed.
"""


# URL for the HathiTrust Extracted Features API (EF 2.0)
BASE_URL = "https://data.htrc.illinois.edu/ef-api/volumes/"

# Rsync module for EF 2.5 (features-2025.04)
# Found here: data.analytics.hathitrust.org::features-2025.04/
RSYNC_BASE = "data.analytics.hathitrust.org::features-2025.04"


# Helper functions

def load_htids(filepath):
    """ read in one HTID per line from a txt file"""
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_pages_csv(htid, pages, output_dir):
    """
    For the final CSV. 
    Write a CSV file with columns: page, words_pos

    The 'words_pos' is body.tokenPosCount for the page.
    Works for both API and rsync-based EF data, as long as
    'pages' is a list of dicts that each have "seq" and "body".
    """
    os.makedirs(output_dir, exist_ok=True)

    #Sanitize the filename.
    safe_htid = htid.replace("/", "+").replace(":", "+")
    output_file = os.path.join(output_dir, f"{safe_htid}.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "words_pos"])

        for page in pages:
            seq = page.get("seq")
            body = page.get("body") or {}
            token_pos_count = body.get("tokenPosCount")

            if token_pos_count is not None:
                writer.writerow([seq, token_pos_count])
            else:
                writer.writerow([seq, "No body data"])

    print(f"Saved → {output_file}")



# API METHOD

def fetch_pages_for_htid_api(htid, session=None):
    """Fetch pages for a volume using the EF API."""
    if session is None:
        session = requests.Session()

    url = f"{BASE_URL}{htid}/pages"
    print(f"Requesting {url}")

    try:
        response = session.get(url)
    except requests.RequestException as e:
        print(f"Error requesting {htid}: {e}")
        return None

    if response.status_code != 200:
        print(f"Failed for {htid}: HTTP {response.status_code}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"Invalid JSON for {htid}")
        return None

    return data.get("data", {}).get("pages", [])


def run_api_pipeline(htid_file, output_dir):
    """Drives the API-based method"""
    htids = load_htids(htid_file)
    print(f"Loaded {len(htids)} htids (API method).")

    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()

    for htid in htids:
        print(f"\nProcessing {htid} with API.")
        pages = fetch_pages_for_htid_api(htid, session=session)

        if pages is None:
            print(f"No pages in {htid}, so skipping this one!")
            continue

        write_pages_csv(htid, pages, output_dir)


# RSYNC METHOD

# From https://github.com/htrc/htrc-feature-reader/blob/master/htrc_features/utils.py
def _id_encode(id_):
    """
    :param id_: The part of the HTID after the library code (after the first '.').
                e.g. for 'vol.123/456', this is '123/456'.
    :return: A sanitized id. e.g., 123/456 -> 123=456 to avoid filesystem issues.
    """
    return id_.replace(":", "+").replace("/", "=").replace(".", ",")


# From https://github.com/htrc/htrc-feature-reader/blob/master/htrc_features/utils.py
def clean_htid(htid):
    """
    :param htid: A HathiTrust ID of form lib.vol; e.g. mdp.1234
    :return: A sanitized version of the HathiTrust ID, appropriate for filename use.
    """
    libid, volid = htid.split(".", 1)
    volid_clean = _id_encode(volid)
    return ".".join([libid, volid_clean])


# From https://github.com/htrc/htrc-feature-reader/blob/master/htrc_features/utils.py
def id_to_stubbytree(htid, format=None, suffix=None, compression=None):
    """Take an HTRC id and convert it to a 'stubbytree' location."""
    libid, volid = htid.split(".", 1)
    volid_clean = _id_encode(volid)

    suffixes = [s for s in [format, compression] if s is not None]
    filename = ".".join([clean_htid(htid), *suffixes])
    #path = os.path.join(libid, volid_clean[::3], filename)
    path = f"{libid}/{volid_clean[::3]}/{filename}"
    return path


def htid_to_rsync_path(htid):
    """Convert an HTID to its EF 2.5 stubbytree path for 
    the .json.bz2 EF file.

    Example:
      htid = 'nyp.33433070251792'
      returns 'nyp/33759/nyp.33433070251792.json.bz2'
    """
    return id_to_stubbytree(htid, format="json", compression="bz2")


def rsync_download_ef(htid, download_dir):
    """
    Use rsync to download the EF .json.bz2 file for a single HTID
    into download_dir. Uses local path for the downloaded file.
    """
    os.makedirs(download_dir, exist_ok=True)

    try:
        rsync_path = htid_to_rsync_path(htid)
    except ValueError as e:
        print(f"Skipping {htid}: {e}")
        return None

    # cmd = [
    #     "rsync", "-av", "--no-relative",
    #     f"{RSYNC_BASE}/{rsync_path}",
    #     download_dir + "/",
    # ]

    remote = f"{RSYNC_BASE}/{rsync_path}"

    local_dest = os.path.abspath(download_dir)
    local_dest = local_dest.replace("\\", "/") + "/"

    # --no-relative so the file goes directly to download_dir
    # instead of recreating whole stubbytree directory
    cmd = [
        "rsync", "-av", "--no-relative",
        remote,
        local_dest,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Rsync failed for {htid}:\n{result.stderr}")
        return None

    local_file = os.path.join(download_dir, os.path.basename(rsync_path))
    if not os.path.exists(local_file):
        print(f"Downloaded file not found for {htid}: {local_file}")
        return None

    return local_file


def load_pages_from_ef(local_bz2_path):
    """
    Open a .json.bz2 EF file and return the list of pages.
    EF 2.0 and 2.5 store page data under features: pages.
    """
    try:
        with bz2.open(local_bz2_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading EF JSON from {local_bz2_path}: {e}")
        return None

    pages = data.get("features", {}).get("pages", [])
    if not isinstance(pages, list):
        print(f"Missing 'pages' in {local_bz2_path}")
        return None

    return pages


def run_rsync_pipeline(htid_file, output_dir):
    """Drives the rsync-based method"""
    htids = load_htids(htid_file)
    print(f"Loaded {len(htids)} htids (rsync method).")

    # Where to stash the raw EF .json.bz2 files
    ef_dir = os.path.join(output_dir, "ef_raw")
    os.makedirs(ef_dir, exist_ok=True)

    for htid in htids:
        print(f"\nProcessing {htid} with rsync.")

        local_bz2 = rsync_download_ef(htid, ef_dir)
        if local_bz2 is None:
            print(f"Skipping {htid} - JSON didn't download.")
            continue

        pages = load_pages_from_ef(local_bz2)
        if pages is None:
            print(f"Skipping {htid} - check the 'pages' structure.")
            continue

        write_pages_csv(htid, pages, output_dir)



def main():
    parser = argparse.ArgumentParser(
        description="Download HathiTrust Extracted Features and export per-volume CSVs."
    )

    parser.add_argument(
        "--htids",
        required=True,
        help="Path to text file containing one HTID per line",
    )

    parser.add_argument(
        "--out",
        default="ef_pages_csv",
        help="Output directory for CSV files (default: ef_pages_csv)",
    )

    parser.add_argument(
        "--method",
        choices=["api", "rsync"],
        required=True,
        help="Which method to use: 'api' or 'rsync'",
    )

    args = parser.parse_args()

    if args.method == "api":
        run_api_pipeline(args.htids, args.out)
    elif args.method == "rsync":
        run_rsync_pipeline(args.htids, args.out)
    else:
        # This shouldn't happen because of argparse choices, but just in case:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
