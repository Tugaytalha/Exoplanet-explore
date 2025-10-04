#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch *all* columns from NASA Exoplanet Archive KOI cumulative and Stellar Hosts
tables, merge them, and compute Sun-centric Cartesian positions.

Outputs (in ./data):
  - koi_cumulative.csv
  - stellarhosts.csv
  - koi_with_relative_location.csv   (full columns + x_pc/y_pc/z_pc/dist_ly)
"""

import os, io, math, re, requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------------

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DATA_DIR     = "data"                   # change if you like

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def tap_csv(query: str) -> pd.DataFrame:
    """
    Run an ADQL query against the Exoplanet Archive TAP service and return a DataFrame.
    We keep the query comment-free to avoid 400 errors.
    """
    r = requests.get(
        TAP_SYNC_URL,
        params={"query": query, "format": "csv"},
        timeout=300,
        headers={"User-Agent": "exoplanet-explore/1.0"},
    )
    if r.status_code != 200:
        raise RuntimeError(f"TAP {r.status_code} error:\n{r.text[:2000]}")
    return pd.read_csv(io.StringIO(r.text))

def host_from_kepler_name(name: str) -> str | None:
    """
    Strip planet suffix: 'Kepler-62 f' → 'Kepler-62'.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    m = re.match(r"^\s*([^\s]+(?:-[^\s]+)?)", name.strip())
    return m.group(1) if m else None

def sph_to_cart_pc(ra_deg: float, dec_deg: float, dist_pc: float):
    """
    Convert (RA, Dec, distance[pc]) → Cartesian parsecs.
    """
    ra  = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = dist_pc * math.cos(dec) * math.cos(ra)
    y = dist_pc * math.cos(dec) * math.sin(ra)
    z = dist_pc * math.sin(dec)
    return x, y, z

# ---------------------------------------------------------------------------
# Downloads (now select * to grab every column)
# ---------------------------------------------------------------------------

def download_koi_cumulative(save_path=f"{DATA_DIR}/koi_cumulative.csv") -> pd.DataFrame:
    ensure_dir(save_path)
    query = "SELECT * FROM cumulative"                 # ← all columns
    df = tap_csv(query)
    df.to_csv(save_path, index=False)
    return df

def download_stellar_hosts(save_path=f"{DATA_DIR}/stellarhosts.csv") -> pd.DataFrame:
    ensure_dir(save_path)
    query = "SELECT * FROM stellarhosts"               # ← all columns
    df = tap_csv(query)
    df.to_csv(save_path, index=False)
    return df

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_relative_locations(koi_df: pd.DataFrame, sh_df: pd.DataFrame) -> pd.DataFrame:
    koi = koi_df.copy()
    koi["host"] = koi["kepler_name"].apply(host_from_kepler_name)

    # Rename RA/Dec in stellar table so we don't collide with KOI columns
    sh = sh_df.rename(columns={"ra": "host_ra", "dec": "host_dec"})

    merged = koi.merge(sh, how="left", left_on="host", right_on="hostname")

    # Prefer stellar-host coordinates when a distance exists
    merged["use_ra"]  = np.where(merged["sy_dist"].notna(), merged["host_ra"], merged["ra"])
    merged["use_dec"] = np.where(merged["sy_dist"].notna(), merged["host_dec"], merged["dec"])

    # Compute 3-D position (rows without distance remain NaN)
    xyz = merged.apply(
        lambda r: sph_to_cart_pc(r["use_ra"], r["use_dec"], r["sy_dist"])
        if pd.notna(r["sy_dist"]) else (np.nan, np.nan, np.nan),
        axis=1, result_type="expand"
    )
    xyz.columns = ["x_pc", "y_pc", "z_pc"]
    merged = pd.concat([merged, xyz], axis=1)

    merged["dist_ly"] = merged["sy_dist"] * 3.26156    # 1 pc = 3.26156 ly
    return merged                                      # keep *all* columns

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    koi     = download_koi_cumulative()
    stellar = download_stellar_hosts()
    merged  = compute_relative_locations(koi, stellar)

    out_path = f"{DATA_DIR}/koi_with_relative_location.csv"
    ensure_dir(out_path)
    merged.to_csv(out_path, index=False)

    print("Saved:")
    print(f" - {DATA_DIR}/koi_cumulative.csv")
    print(f" - {DATA_DIR}/stellarhosts.csv")
    print(f" - {out_path}")
    print("\nPreview (first 5 rows with known distance):")
    print(merged[merged["sy_dist"].notna()].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
