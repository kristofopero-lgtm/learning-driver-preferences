#### Doel van dit script: te weten komen of de taskid die in de jsonfiles per folder gebruikt worden (dus binnen eenzelfde dag/route) altijd naar dezelfde route-coordinaten verwijzen, dus binnen 1 route/day. ####
#### Antwoord: meestal wel, maar er zijn afwijkingen, in de meeste folders is er geen afwijking maar in sommige wel. Wat betekent dat? De afwijking lijkt niet zo groot, als ze er is, maar zou je moeten nakijken. En vooral eerst vragen hoe dat komt: door het normaliseren? of...?  ####
### Verder te onderzoeken of niet van belang?? ###

from pathlib import Path
import re
import pandas as pd
import numpy as np

from index_files import REQUESTS_DIR, OUTPUT, parse_request_file, parse_time_from_filename
from monitor_taskids import split_depot_region_date


def collect_rows_per_folder(root: Path):
    """
    Walk one level below 'root' and collect rows:
    [folder, file, task_id, lat, lon, from, till, hhmmss, depot_region, date]
    """
    rows = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        prov = split_depot_region_date(folder.name)
        for p in folder.glob("*.json"):
            parsed = parse_request_file(p)
            tasks = parsed.get("tasks", [])
            if not tasks:
                continue
            hhmmss = parse_time_from_filename(p.name)
            for t in tasks:
                tid = t.get("id")
                lat = t.get("lat")
                lon = t.get("lon")
                if tid is None or pd.isna(lat) or pd.isna(lon):
                    continue
                rows.append({
                    "folder": folder.name,             # e.g., '0521_300-20220617'
                    "file": str(p.relative_to(root)),  # relative path for provenance
                    "task_id": tid,
                    "lat": float(lat),
                    "lon": float(lon),
                    "from": t.get("from"),
                    "till": t.get("till"),
                    "hhmmss": hhmmss,                  # optional intra-day order hint
                    "depot_region": prov["depot_region"],
                    "date": prov["date"],
                })
    return pd.DataFrame(rows)

# --------- consistency test per folder/day ---------

def test_per_folder_consistency(df_all: pd.DataFrame, round_decimals=None):
    """
    Check invariant within each folder/day:
      A (folder, task_id) must map to exactly one (lat, lon) across all files in that folder.

    Parameters:
      round_decimals: Optional int. If you discover minor float noise, pass e.g. 8 or 10.

    Returns:
      df_unique       : unique (folder, task_id, lat, lon) with full provenance (files list)
      df_inconsistent : rows for (folder, task_id) that have >1 distinct (lat, lon)
      df_summary      : per-folder counts and inconsistency rate
    """
    if df_all.empty:
        raise RuntimeError("No task rows found under REQUESTS_DIR. Check files/structure.")

    # Optional numeric normalization if needed later
    if isinstance(round_decimals, int) and round_decimals > 0:
        df_all["lat"] = df_all["lat"].round(round_decimals)
        df_all["lon"] = df_all["lon"].round(round_decimals)

    # Unique (folder, task_id, lat, lon), preserving provenance
    df_unique = (df_all
                 .groupby(["folder", "task_id", "lat", "lon"], as_index=False)
                 .agg(files=("file", lambda s: sorted(set(s))),
                      hhmmss_list=("hhmmss", lambda s: [x for x in s if x]),  # optional time hints
                      appearances=("file", "size")))

    # Count distinct lat/lon per (folder, task_id)
    counts = (df_unique
              .groupby(["folder", "task_id"], as_index=False)
              .size()
              .rename(columns={"size": "unique_latlon_count"}))

    # Inconsistencies: same (folder, task_id) appears with >1 coordinates within the day
    df_inconsistent = counts[counts["unique_latlon_count"] > 1].copy()

    # Per-folder summary
    df_summary = (counts
                  .groupby("folder", as_index=False)
                  .agg(
                      task_ids=("task_id", "nunique"),
                      inconsistent_task_ids=("unique_latlon_count", lambda s: (s > 1).sum())
                  ))
    df_summary["inconsistency_rate"] = (
        df_summary["inconsistent_task_ids"] / df_summary["task_ids"]
    ).round(4)

    # For convenience: attach the coordinate details to the inconsistent pairs
    if not df_inconsistent.empty:
        # Build a map of (folder, task_id) -> list of (lat, lon, files)
        details = (df_unique
                   .groupby(["folder", "task_id"], as_index=False)
                   .agg(locations=("lat", lambda s: list(zip(s, df_unique.loc[s.index, "lon"]))),
                        files_per_location=("files", list)))
        df_inconsistent = df_inconsistent.merge(details, on=["folder", "task_id"], how="left")

    return df_unique, df_inconsistent, df_summary

# --------- runner ---------

def main():
    root = REQUESTS_DIR
    out = OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    df_all = collect_rows_per_folder(root)
    df_unique, df_inconsistent, df_summary = test_per_folder_consistency(df_all)

    # Save reports
    df_unique.to_csv(out / "per_folder_taskid_locations.csv", index=False)
    df_inconsistent.to_csv(out / "per_folder_inconsistencies.csv", index=False)
    df_summary.to_csv(out / "per_folder_summary.csv", index=False)

    # Console summary
    n_folders = df_all["folder"].nunique()
    print(f"Folders scanned: {n_folders}")
    print(f"Unique (folder, task_id, lat, lon): {len(df_unique)}  → {out / 'per_folder_taskid_locations.csv'}")
    print(f"Inconsistent (folder, task_id) pairs: {len(df_inconsistent)} → {out / 'per_folder_inconsistencies.csv'}")
    print("Per-folder summary (first 10):")
    print(df_summary.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
