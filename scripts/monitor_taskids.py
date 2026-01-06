#### Doel van dit script: te weten komen of de taskid die in de verschillende jsonfiles gebruikt worden altijd naar dezelfde route-coordinaten verwijzen, dus over de routes/days heen. ####
#### Antwoord: nee. Eenzelfde taskid heeft niet over alle files heen dezelfde lat/lon. Er zijn best wel wat taskids met verschillende lat/lon over de files heen. ####
### Dus: nagaan of binnen elke folder (route/day) elke taskid wel dezelfe coordinaten heeft. Zie monitor_taskids_per_folder.py ###

# Import variables and functions created in index_files
from index_files import HERE, BASE_DIR, DATA_DIR, REQUESTS_DIR, RESPONSES_DIR, OUTPUT, parse_request_file

from pathlib import Path
import pandas as pd

####################################
# HELPER FUNCTIONS
####################################

# To keep track of the provenance of the taskid - split the foldername into depot_region and date
def split_depot_region_date(folder_name: str):
    if "-" in folder_name:
        left, right = folder_name.split("-", 1)  # only first hyphen
        return {"depot_region": left, "date": right}
    return {"depot_region": folder_name, "date": ""}

####################################
# BASE FUNCTION: BUILD TASKID-DF
####################################
def build_taskid_latlon_df(path: Path, keep_provenance=True):
    records = []

    # to define provenance of the taskids in the df
    for folder in path.iterdir():
        if not folder.is_dir():
            continue
        provenance = split_depot_region_date(folder.name) if keep_provenance else {}

        # parse
        for p in folder.glob("*.json"):
            parsed = parse_request_file(p)
            tasks = parsed.get("tasks", [])
            if not tasks:
                continue

            for t in tasks:
                tid, lat, lon = t.get("id"), t.get("lat"), t.get("lon")
                if tid is None or pd.isna(lat) or pd.isna(lon):
                    continue
                records.append({
                    "file": str(p.relative_to(path)),
                    "task_id": tid,
                    "lat": lat,
                    "lon": lon,
                    "folder": folder.name,      # e.g., '0521_300-20220617'
                    **provenance                       # depot_region | date
                })

    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise RuntimeError("No task rows found under 'requests/'. Check files/structure.")

    # One row per unique (task_id, lat, lon) while preserving full provenance
    df_unique = (df_all
                 .groupby(["task_id", "lat", "lon"], as_index=False)
                 .agg(
                     files=("file", lambda s: sorted(set(s))),
                     folders=("folder", lambda s: sorted(set(s))),
                     depot_regions=("depot_region", lambda s: sorted(set(s))) if keep_provenance and "depot_region" in df_all.columns else ("file", lambda s: []),
                     dates=("date", lambda s: sorted(set(s))) if keep_provenance and "date" in df_all.columns else ("file", lambda s: []),
                     appearances=("file", "size"),
                 ))

    # Count unique lat/lon per task_id (detect movers)
    counts = df_unique.groupby("task_id").size().reset_index(name="unique_latlon_count")

    # Movers: task_ids with >1 distinct (lat, lon)
    movers = counts[counts["unique_latlon_count"] > 1].copy()

    return df_unique, movers

####################################
# RUN THE BUILD_TASKID
####################################

if __name__ == "__main__":
    df_result, movers = build_taskid_latlon_df(REQUESTS_DIR, keep_provenance=True)

    # Save
    df_result.to_csv(OUTPUT / "taskid_latlon_unique.csv", index=False)
    movers.to_csv(OUTPUT / "taskids_with_multiple_locations.csv", index=False)

    print("Unique (task_id, lat, lon):", len(df_result))
    print("Task IDs with >1 unique lat/lon:", len(movers))
    print(df_result.head(10).to_string(index=False))
