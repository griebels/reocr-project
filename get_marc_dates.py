import re
import sys
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import requests
import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import quote

def get_htid_list(csv_path, col_name):
    df = pd.read_csv(csv_path, encoding='utf-8', dtype=str)
    if col_name not in df.columns:
        print(f"Error: HTID column needs to be {col_name}!")
        sys.exit(1)
    entries = []
    for val in df[col_name].fillna(""):
        s = str(val).strip()
        entries.append(s)
    return entries

# Helper function for datetime
def date_to_iso(tt):
    """
    Try some ISO formats to see if we can get it to be proper datetime.
    If the format is "20020606093309.7" (i.e., the end of
    the fraction is not .0) the function will round the 
    seconds up or down depending on the
    fraction, and then will convert it to an ending of ".0".

    Otherwise, if none of these options works, return 
    the original string.
    """
    if not tt:
        return np.nan
    tt = str(tt).strip()
    
    # Match strings like 20230925123456.0–9
    m = re.match(r"^(\d{14})\.(\d)$", tt)
    if m:
        base, frac = m.groups()
        try:
            dt = datetime.datetime.strptime(base, "%Y%m%d%H%M%S")
            frac_integer = int(frac)
            if frac_integer >= 5:
                dt +=datetime.timedelta(seconds=1) #round it up
            return dt.isoformat()
        except Exception:
            pass

    for formt in ("%Y%m%d%H%M%S.0", "%Y-%m-%dT%H:%M:%SZ", "%Y%m%d"): #not sure if still need "%Y%m%d%H%M%S.0" but leaving it in just inc ase.
        try:
            dt = datetime.datetime.strptime(tt, formt)
            return dt.isoformat() + ("Z" if formt.endswith("Z") else "") #just carrying over that Z in case it's useful
        except Exception:
            pass

    #and if all else fails, just return the original string
    return tt

def do_blank_row(htid):
    return{
        "htid":htid,
        "link": f"https://babel.hathitrust.org/cgi/pt?id={htid}",
        "field_005": "NA",
        "field_974d": "NA",
        "974_group": "NA",
        # "DAT" : "NA" # DAT fields will be added dynamically
    }

def fetch_marc_json(htid, session: requests.Session, timeout: float=20.0):
    # url encode in case htid has unsafe characters
    #url_htid=quote(htid, safe="")
    #url = f"https://catalog.hathitrust.org/api/volumes/full/htid/{url_htid}.json"
    url = f"https://catalog.hathitrust.org/api/volumes/full/htid/{htid}.json"
    
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

# NS = {"m": "http://www.loc.gov/MARC21/slim"} #for MARC xml namespace mapping

def extract_marc_fields(marc_xml, htid):
    root = ET.fromstring(marc_xml)
    out = do_blank_row(htid)

    # field 005
    # confld005 = root.find(".//m:controlfield[@tag='005']", NS)
    confld005 = root.find(".//{http://www.loc.gov/MARC21/slim}controlfield[@tag='005']")
    if confld005 is not None and confld005.text:
        out["field_005"] = date_to_iso(confld005.text.strip())
    
    # DAT fields
    for df in root.findall(".//{http://www.loc.gov/MARC21/slim}datafield[@tag='DAT']"):
        ind1 = (df.get("ind1") or "")
        for sf in df.findall(".//{http://www.loc.gov/MARC21/slim}subfield"):
            code = sf.get("code")
            if code:
                if ind1:
                    key = f"DAT{ind1}{code}"
                else:
                    key = f"DAT{code}"
                val = (sf.text or "").strip()
                out[key] = date_to_iso(val) if val else np.nan
    
    # 974 group or single
    all_974 = root.findall(".//{http://www.loc.gov/MARC21/slim}datafield[@tag='974']")
    if len(all_974) ==0:
        out['974_group']= "None"
    elif len(all_974)==1:
        out['974_group']= 'Single'
    else:
        out['974_group']= "Multi"

    # 974d for this HTID
    #matched_974d = "NA"
    for df in all_974:
        u_sf = [(sf.text or "").strip() for sf in df.findall(".//{http://www.loc.gov/MARC21/slim}subfield[@code='u']")]
        if htid in u_sf:
            d_sf = df.find(".//{http://www.loc.gov/MARC21/slim}subfield[@code='d']")
            if d_sf is not None and d_sf.text:
                raw_dtime = d_sf.text.strip()
                out["field_974d"] = date_to_iso(raw_dtime)
            break

    return out

def order_columns(df):
    front = ['htid', 'link', 'field_005', 'field_974d', '974_group']
    dat_cols = sorted([c for c in df.columns if c.startswith("DAT")])
    others = [c for c in df.columns if c not in set(front + dat_cols)]
    ordered = [c for c in front if c in df.columns] + dat_cols + sorted(others)
    return df[ordered]

# THE MAIN PROCESSING BELOW ######
def process_csv(csv_path, out_csv, source_col):
    entries = get_htid_list(csv_path, source_col)
    total = len(entries)
    sess = requests.Session()

    rows: List[Dict[str, Any]] = []
    for idx, htid in enumerate(entries, 1):
        # progress report every 1000 rows (and at the end)
        if idx % 1000 == 0 or idx == total:
            print(f"processed {idx} rows")
    # for i in entries:
    #     htid=i

        row= do_blank_row(htid)
        data = fetch_marc_json(htid, sess)
        if not data:
            rows.append(row)
            continue

        recs = data.get("records", {})
        if not recs:
            rows.append(row)
            continue

        first_rec = next(iter(recs.values()))
        marc_xml = first_rec.get("marc-xml","")
        if marc_xml:
            parsed = extract_marc_fields(marc_xml, htid)
            row= parsed
    
        rows.append(row)
    
    df = order_columns(pd.DataFrame(rows))
    # if len(df)% 1000 == 0:
    #     print(f"processed {len(df)} rows.")

    df.to_csv(out_csv, index=False,)
    print(f"Saved {len(df)} rows to {out_csv}")

# AND FINALLY... ######
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch only 005, DAT, and 974d fields for HTIDs in a CSV.")
    parser.add_argument("csv_path", help="Path to input CSV")
    parser.add_argument("--src_col", default="htid", help="Column name that contains HTIDs (default: htid)")
    parser.add_argument("--out", default="marc_output.csv", help="Output CSV path (default: marc_output.csv)")
 
    args = parser.parse_args()
    process_csv(
        csv_path=args.csv_path,
        out_csv=args.out,
        source_col=args.src_col,
    )

    
