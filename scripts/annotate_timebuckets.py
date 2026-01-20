from __future__ import annotations

import pandas as pd
from datetime import datetime, time
from pathlib import Path

from learning_driver_preferences.paths import OUTPUT

# Set up timeframes
# Indien geen second_optimalization_round: zet tuple op (time(0, 0), time(0, 0))
# En dan: first_optimalization_round:  (time(5, 0),  time(11, 0))
first_optimalisation_round = (time(5, 0),  time(11, 0))
second_optimalisation_round = (time(0, 0), time(0, 0))
not_relevant = (time(11, 1), time(23, 59))

########################################################################
# HELPER FUNCTIONS
########################################################################

# To define if a time is within a timeframe
def in_timeframe(t: time, timeframe: tuple[time,time]) -> bool:
    return t is not None and timeframe[0] <= t < timeframe[1]

# To classify the timeframe of the request-file (request was made when?)
def classify_request_timeframe(time_in_isoformat: str) -> str:
    if pd.isna(time_in_isoformat): return "unknown"
    time = datetime.strptime(time_in_isoformat, "%H:%M:%S").time()
    if in_timeframe(time, first_optimalisation_round): return "first_round"
    if in_timeframe(time, second_optimalisation_round): return "second_round"
    if in_timeframe(time, not_relevant): return "not_relevant"
    return "unknown"


########################################################################
# BASE FUNCTION: ANNOTATE TIME BUCKETS AND INFORMATION ABOUT REQUESTS (RUN_NUMBER AND RUN_LABEL)
########################################################################

def annotate_time_buckets(input_csv, out_csv):

    df_files = pd.read_csv(input_csv)

    # Ensure required columns exist
    required = {"route_id", "date", "request_time"}
    missing = required - set(df_files.columns)
    if missing:
        raise KeyError(f"Missing required columns in {input_csv.name}: {sorted(missing)}")

    # Compute time buckets from request_time strings
    df_files["time_bucket"] = df_files["request_time"].apply(classify_request_timeframe)

   # Parse request_time and compute minute_of_day - to order the request_times in an easy way, minute_of_day gives an integer from 0 - 1439 (24x60min in a day), and you keep order across the day
    t = pd.to_datetime(df_files["request_time"], format="%H:%M:%S", errors="coerce")
    df_files["minute_of_day"] = t.dt.hour*60 + t.dt.minute

    # Number runs within each (route_id, date, time_bucket) (run_number)
    # Only for relevant buckets (first_round / second_round).
    relevant = df_files["time_bucket"].isin(["first_round", "second_round"])
    df_files["run_number"] = pd.NA

    # Compute run_number by chronological order per (route_id, date, time_bucket)
    df_files.loc[relevant, "run_number"] = (
        df_files.loc[relevant]
                .sort_values(["route_id","date","minute_of_day"])
                .groupby(["route_id","date","time_bucket"])
                .cumcount() + 1
    )

    # Determine group sizes to identify 'last_run'
    group_sizes = (
        df_files[relevant]
        .groupby(["route_id","date","time_bucket"])["request_time"]
        .transform("size")
    )
    df_files.loc[relevant, "group_size"] = group_sizes

    # Assign run_label based on run_number and group_size
    df_files["run_label"] = "not_relevant"
    df_files.loc[df_files["time_bucket"].eq("unknown"), "run_label"] = "unknown"

    condition_only  = relevant & (df_files["run_number"].astype("Int64") == 1) & (df_files["group_size"] == 1)
    condition_first = relevant & (df_files["run_number"].astype("Int64") == 1) & (df_files["group_size"] > 1)
    condition_last  = relevant & (df_files["run_number"].astype("Int64") == df_files["group_size"]) & (df_files["group_size"] > 1)
    condition_mid   = relevant & ~(condition_only | condition_first | condition_last)

    df_files.loc[condition_only,  "run_label"] = "only_run"
    df_files.loc[condition_first, "run_label"] = "first_run"
    df_files.loc[condition_last,  "run_label"] = "last_run"
    df_files.loc[condition_mid,   "run_label"] = "intermediate_run"

    # Set run_number for non-relevant rows to NA
    df_files.loc[~relevant, "run_number"] = pd.NA

    # Persist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_files.to_csv(out_csv, index=False)
    return out_csv

########################################################################
# RUN ANNOTATE_TIMEBUCKETS
########################################################################

if __name__ == "__main__":
    result_path = annotate_time_buckets(
        input_csv=OUTPUT / "index_files.csv",
        out_csv=OUTPUT / "index_files_with_time_buckets.csv"
    )
    print(f"Annotated file written to: {result_path.resolve()}")
