#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, math, re, requests
import pandas as pd
import numpy as np

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ---------- Helpers ----------

def tap_csv(query: str) -> pd.DataFrame:
    """
    Run a TAP (ADQL) query against NASA Exoplanet Archive and return a DataFrame.
    Use GET with ?query=...&format=csv per docs.
    """
    # TIP: keep the query comment-free; ADQL comments often trigger 400.
    r = requests.get(
        TAP_SYNC_URL,
        params={"query": query, "format": "csv"},
        timeout=180,
        headers={"User-Agent": "exoplanet-explore/1.0"}
    )
    if r.status_code != 200:
        # show server message to make debugging easy
        raise RuntimeError(f"TAP {r.status_code} error:\n{r.text[:2000]}")
    return pd.read_csv(io.StringIO(r.text))

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def host_from_kepler_name(name: str) -> str | None:
    """
    'Kepler-62 f' -> 'Kepler-62'. Returns None if name is empty.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    m = re.match(r"^\s*([^\s]+(?:-[^\s]+)?)", name.strip())
    return m.group(1) if m else None

def sph_to_cart_pc(ra_deg: float, dec_deg: float, dist_pc: float):
    """(RA,Dec,dist_pc) -> (x,y,z) in parsecs."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = dist_pc * math.cos(dec) * math.cos(ra)
    y = dist_pc * math.cos(dec) * math.sin(ra)
    z = dist_pc * math.sin(dec)
    return x, y, z

# ---------- Downloads ----------

def download_koi_cumulative(save_path="data/koi_cumulative.csv") -> pd.DataFrame:
    # Valid table name is 'cumulative' (KOI Cumulative Delivery). :contentReference[oaicite:1]{index=1}
    koi_query = (
        "SELECT kepid, kepoi_name, kepler_name, ra, dec "
        "FROM cumulative"
    )
    df = tap_csv(koi_query)
    ensure_dir(save_path)
    df.to_csv(save_path, index=False)
    return df

def download_stellar_hosts(save_path="data/stellarhosts.csv") -> pd.DataFrame:
    # 'stellarhosts' contains hostname, ra/dec and sy_dist (pc). :contentReference[oaicite:2]{index=2}
    sh_query = (
        "SELECT hostname, ra AS host_ra, dec AS host_dec, sy_dist "
        "FROM stellarhosts "
        "WHERE sy_dist IS NOT NULL"
    )
    df = tap_csv(sh_query)
    ensure_dir(save_path)
    df.to_csv(save_path, index=False)
    return df

# ---------- Processing ----------

def compute_relative_locations(koi_df: pd.DataFrame, sh_df: pd.DataFrame) -> pd.DataFrame:
    koi = koi_df.copy()
    koi["host"] = koi["kepler_name"].apply(host_from_kepler_name)

    merged = koi.merge(sh_df, how="left", left_on="host", right_on="hostname")

    # Use host coordinates when we have a distance, else fall back on KOI ra/dec for display
    merged["use_ra"]  = np.where(merged["sy_dist"].notna(), merged["host_ra"], merged["ra"])
    merged["use_dec"] = np.where(merged["sy_dist"].notna(), merged["host_dec"], merged["dec"])

    xyz = merged.apply(
        lambda r: sph_to_cart_pc(r["use_ra"], r["use_dec"], r["sy_dist"])
        if pd.notna(r["sy_dist"]) else (np.nan, np.nan, np.nan),
        axis=1, result_type="expand"
    )
    xyz.columns = ["x_pc", "y_pc", "z_pc"]
    out = pd.concat([merged, xyz], axis=1)
    out["dist_ly"] = out["sy_dist"] * 3.26156

    keep = [
        "kepid","kepoi_name","kepler_name","host","hostname",
        "ra","dec","host_ra","host_dec","use_ra","use_dec",
        "sy_dist","dist_ly","x_pc","y_pc","z_pc"
    ]
    return out[keep]

def main():
    koi = download_koi_cumulative()
    stellar = download_stellar_hosts()
    out = compute_relative_locations(koi, stellar)

    ensure_dir("data/koi_with_relative_location.csv")
    out.to_csv("data/koi_with_relative_location.csv", index=False)

    print("Saved files:")
    print(" - data/koi_cumulative.csv")
    print(" - data/stellarhosts.csv")
    print(" - data/koi_with_relative_location.csv")
    print("\nPreview with known distances:")
    print(out[out["sy_dist"].notna()].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
