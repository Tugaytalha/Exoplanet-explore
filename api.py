# main.py
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

DATA_PATH = Path("data/koi_with_relative_location.csv")

# ---------- load & prepare the table once at startup ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found – run fetch.py first to create it."
    )

df = pd.read_csv(DATA_PATH)

# Dummy label: any row with a known distance is treated as an exoplanet
df["is_exoplanet"] = df["sy_dist"].notna()

# -------- FastAPI app -----------
app = FastAPI(
    title="Kepler KOI Relative-Position API",
    description="Serves Sun-centric Cartesian positions for KOIs "
                "along with a dummy `is_exoplanet` flag.",
    version="0.1.0",
)

# helper – remap a dataframe row to the JSON we want to expose
def row_to_dict(row: pd.Series) -> dict:
    return {
        "kepid":         int(row.kepid),
        "kepoi_name":    row.kepoi_name,
        "kepler_name":   row.kepler_name,
        "x_pc":          None if pd.isna(row.x_pc) else float(row.x_pc),
        "y_pc":          None if pd.isna(row.y_pc) else float(row.y_pc),
        "z_pc":          None if pd.isna(row.z_pc) else float(row.z_pc),
        "is_exoplanet":  bool(row.is_exoplanet),
    }

# -------- endpoints --------
@app.get("/planets", response_model=List[dict])
def list_planets(
    skip: int = Query(0, ge=0, description="Rows to skip (pagination start)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum rows to return"),
    only_exoplanet: Optional[bool] = Query(
        None, description="If true, return only rows with is_exoplanet=True"
    ),
):
    """
    Stream KOI rows with Cartesian coordinates.\n
    Use `skip` & `limit` for pagination.  
    Optional `only_exoplanet=true` to filter.
    """
    subset = df
    if only_exoplanet is True:
        subset = subset[subset["is_exoplanet"]]

    rows = subset.iloc[skip : skip + limit]
    return [row_to_dict(r) for _, r in rows.iterrows()]


@app.get("/planets/{kepid}", response_model=dict)
def get_planet(kepid: int):
    """
    Return a single KOI by its Kepler ID (kepid).
    """
    rows = df[df["kepid"] == kepid]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Kepler ID {kepid} not found in table.",
        )
    return row_to_dict(rows.iloc[0])
