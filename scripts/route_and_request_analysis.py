from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# YOUR HELPERS (as provided, unchanged)
# ============================================================
def read_inputfile(
    inputfile: str | Path,
    depot: Optional[str] = None,
):
    # Read CSV or Excel, normalize headers, and (optionally) filter by depot.
    # Standardized columns (renamed when present):
    #   - route_id
    #   - request_time
    #   - depot (derived from route_id if available)
    #   - date
    #   - config_name
    #   - num_tasks
    #   - num_fixed

    # Excel-specific renames (case-insensitive):
    #   NumberOfTasks            -> num_tasks
    #   Date                     -> date
    #   TriggerType              -> config_name
    #   NumberOfTasksInInputPlan -> num_fixed
    #   RouteId                  -> route_id
    #   Time                     -> request_time

    # Returns:
    #   df_norm: DataFrame with normalized names and optional depot filtering
    #   info:    basic info dict with metadata (selected_depot, source, filename)

    path = Path(inputfile)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    is_csv = suffix == ".csv"
    is_excel = suffix in {".xlsx", ".xlsm", ".xls"}

    if not (is_csv or is_excel):
        raise ValueError("Unsupported file type. Provide .csv or .xlsx/.xlsm/.xls.")

    if is_csv:
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine="openpyxl")

    # Normalize header whitespace
    df = df.rename(columns={c: c.strip() if isinstance(c, str) else c for c in df.columns})

    # Case-insensitive mapping helper
    lower_map = {str(c).lower(): c for c in df.columns}

    # Shared rename map
    rename_pairs = {}

    # Route/time
    if "routeid" in lower_map:
        rename_pairs[lower_map["routeid"]] = "route_id"
    if "time" in lower_map:
        rename_pairs[lower_map["time"]] = "request_time"

    # Excel column renames (apply when present, CSV included too if headers match)
    if "numberoftasks" in lower_map:
        rename_pairs[lower_map["numberoftasks"]] = "num_tasks"
    if "date" in lower_map:
        rename_pairs[lower_map["date"]] = "date"
    if "configurationname" in lower_map:
        rename_pairs[lower_map["configurationname"]] = "config_name"
    if "numberoftasksininputplan" in lower_map:
        rename_pairs[lower_map["numberoftasksininputplan"]] = "num_fixed"

    # If CSV already has canonical names (e.g., num_tasks), this is a NOOP
    df = df.rename(columns=rename_pairs)

    # Derive depot (first 4 chars of route_id) if route_id exists
    if "route_id" in df.columns:
        df["depot"] = df["route_id"].astype(str).str.strip().str[:4]
    else:
        # If no route_id, keep depot as provided (filtering will be skipped)
        if "depot" not in df.columns:
            df["depot"] = np.nan

    # Optional depot filter (only if depot info is available)
    if depot is not None and "depot" in df.columns:
        df = df.loc[df["depot"].astype(str) == str(depot)].copy()
        selected_depot = str(depot)
    else:
        # Indicate selection in metadata
        selected_depot = "ALL" if df["depot"].notna().any() else "N/A"

    # Normalize 'date' if present: try to keep an ISO string (YYYY-MM-DD)
    if "date" in df.columns:
        # Accept 'YYYYMMDD', datetime, Excel serials, etc.
        d_raw = df["date"]
        if pd.api.types.is_integer_dtype(d_raw) or pd.api.types.is_float_dtype(d_raw):
            # Could be Excel serial or YYYYMMDD as number
            # Try Excel serial: heuristic — serials are typically < 60000 for 21st century
            s = pd.to_datetime(d_raw, errors="coerce", origin="1899-12-30", unit="D")
            # If that failed for many, try YYYYMMDD
            if s.isna().mean() > 0.5:
                s = pd.to_datetime(d_raw.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            # Strings like '20240521' or already ISO
            s = pd.to_datetime(d_raw.astype(str), errors="coerce")
            # Secondary try for compact YYYYMMDD
            mask_bad = s.isna()
            if mask_bad.any():
                s2 = pd.to_datetime(d_raw[mask_bad].astype(str), format="%Y%m%d", errors="coerce")
                s = s.mask(mask_bad, s2)
        df["date"] = s.dt.date.astype(str)  # ISO-like "YYYY-MM-DD", 'NaT'->'NaT' string if NaN

    info = {
        "selected_depot": selected_depot,
        "source": "csv" if is_csv else "excel",
        "filename": path.name,
    }

    print(f"file information: {info}")

    return df, info

def parse_request_times(df: pd.DataFrame):
    # parses df['request_time'] if present. Returns a Series[datetime64[ns]]
    if "request_time" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")

    time_str = df["request_time"].astype(str).str.strip()
    parsed_time = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, format="%I:%M:%S.%f %p", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, format="%I:%M:%S %p", errors="coerce")
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_str, errors="coerce")
    return parsed_time


# ===============================================================================================================
# ANALYSIS HELPERS: MINUTE_OF_DAY, FILTER REQUESTS ON CUTOFF TIME AND TASKS, CREATE A TABLE WITH 1 ROW PER TRIP
# ===============================================================================================================
def ensure_minute_of_day(df: pd.DataFrame):
    """Return a Series with minutes after midnight for each row if possible.
    Tries, in order:
      - 'minute_of_day' column if present and numeric
      - parse_request_times and compute hour*60 + minute
    """
    if "minute_of_day" in df.columns:
        s = pd.to_numeric(df["minute_of_day"], errors="coerce")
        if s.notna().any():
            return s

    t = parse_request_times(df)
    if t.empty or t.isna().all():
        return pd.Series(np.nan, index=df.index)
    return (t.dt.hour * 60 + t.dt.minute).astype(int)

def filter_on_cutoff_time_and_tasks(
 # the combination of route_id and day/date = "a trip" (NL = "rit") = the execution of that route/route_id on a particular day
    df: pd.DataFrame,
    cutoff_minutes: int = 660,      # 11:00
    min_mean_tasks: float = 30.0    # keep only trips (route/day) with mean tasks > 30 (mean tasks = mean of num_tasks across the requests for that trip)
) -> pd.DataFrame:
    """Apply first-round filter (if column exists) and time cutoff; keep mornings with mean tasks > threshold."""
    # Standardize key columns (they may or may not be present depending on source)
    if "date" not in df.columns:
        raise ValueError("Input must include a 'date' column after normalization.")
    if "route_id" not in df.columns:
        raise ValueError("Input must include a 'route_id' column after normalization.")
    if "config_name" not in df.columns:
        raise ValueError("Input must include a 'config_name' column after normalization.")
    if "num_tasks" not in df.columns:
        raise ValueError("Input must include a 'num_tasks' column after normalization.")

    df_filter = df.copy()
    # Filter on cutoff time (default 11h) using minute-of-day
    minutes = ensure_minute_of_day(df_filter)
    df_filter["_minute_of_day"] = minutes  # of: df_f = df_f.assign(_minutes=minutes)
    df_filter = df_filter.loc[df_filter["_minute_of_day"] <= cutoff_minutes]

    # Set num_tasks to numeric - defensive code to be sure num_tasks is numeric
    df_filter["num_tasks"] = pd.to_numeric(df_filter["num_tasks"], errors="coerce")

    # Keep only trips (route_id, date) with mean tasks > threshold
    key = ["route_id", "date"]
    trip_mean_tasks = (
        df_filter.groupby(key)["num_tasks"].mean().reset_index(name="mean_num_tasks_of_trip")
    )
    df_filter_trips = trip_mean_tasks.loc[trip_mean_tasks["mean_num_tasks_of_trip"] > min_mean_tasks, key] # only keep route_id and day, throw away the mean_num_tasks
    df_keep = df_filter.merge(df_filter_trips, on=key, how="inner")

    # Provide weekday
    df_keep = df_keep.drop(columns=["_minute_of_day"], errors="ignore")
    df_keep["date"] = pd.to_datetime(df_keep["date"], errors="coerce")
    df_keep["weekday"] = df_keep["date"].dt.day_name()

    return df_keep

def aggregate_trip(df_keep: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered rows from df_keep (= 1 row per kept request) to a table with trips ( = 1 row per trip = 1 row per route/day; requests are summed acroos the trip)
    Returns columns:
      - route_id, date, weekday
      - requests_count
      - day_mean_tasks
      - estimate_pct, create_pct, add_pct (shares within morning)
      - first_minute, last_minute, span_minutes
    """
    key = ["route_id", "date"] # trip

    # Request counts per trip
    request_counts = df_keep.groupby(key).size().reset_index(name="requests_count")

    # Mean of num_tasks per trip
    trip_mean_tasks = (
        df_keep.groupby(key)["num_tasks"].mean().reset_index(name="mean_tasks_trip")
    )

    # Config_name shares per trip
    # Computes the number of config_name (= type of request = EstimateTime, CreateSequence, AddToSequence) per trip (route, day)
    type_of_requests_count = (
        df_keep.groupby(key + ["config_name"]).size()
        .unstack("config_name", fill_value=0) # unstack = set config_names to separate columns instead of 1 column config_name with types of requests as values in rows
    )

    # Ensure consistent columns
    for config in ["EstimateTime", "CreateSequence", "AddToSequence"]:
        if config not in type_of_requests_count.columns:
            type_of_requests_count[config] = 0

    type_of_requests_count = type_of_requests_count[["EstimateTime", "CreateSequence", "AddToSequence"]]
    type_of_requests_count["total_count"] = type_of_requests_count.sum(axis=1).replace(0, np.nan)
    type_of_requests_portion= type_of_requests_count.div(type_of_requests_count["total_count"], axis=0) * 100.0
    type_of_requests_portion = type_of_requests_portion.rename(
        columns={
            "EstimateTime": "estimate_pct",
            "CreateSequence": "create_pct",
            "AddToSequence": "add_pct",
            "total_count": "total_pct"
        }
    ).reset_index()

    # Merge
    trip_table = (
        request_counts.merge(trip_mean_tasks, on=key, how="left")
        .merge(type_of_requests_count, on=key, how = "left")
        .merge(type_of_requests_portion, on=key, how="left")
        .merge(df_keep[key + ["weekday"]].drop_duplicates(), on=key, how="left")
    )

    return trip_table

# ============================================================
# (2) REQUESTS METRICS
# ============================================================

def basic_information(df: pd.DataFrame, min_mean_tasks: int = 30):
    df_keep_all_tasks = filter_on_cutoff_time_and_tasks(df, min_mean_tasks=0)
    df_keep_with_min_tasks = filter_on_cutoff_time_and_tasks(df, min_mean_tasks=min_mean_tasks)

    number_of_routes_before_filter = df_keep_all_tasks["route_id"].nunique()
    number_of_routes_after_filter  = df_keep_with_min_tasks["route_id"].nunique()

    number_of_days_observed = df_keep_all_tasks["date"].nunique()

    number_of_trips_before_filter = (
        df_keep_all_tasks[["route_id", "date"]].drop_duplicates().shape[0]
    )
    number_of_trips_after_filter = (
        df_keep_with_min_tasks[["route_id", "date"]].drop_duplicates().shape[0]
    )

    number_requests_no_filter = df.drop_duplicates(subset=["route_id","date","request_time"]).shape[0]
    number_requests_after_time = df_keep_all_tasks.drop_duplicates(subset=["route_id","date","request_time"]).shape[0]
    number_requests_after_tasks = df_keep_with_min_tasks.drop_duplicates(subset=["route_id","date","request_time"]).shape[0]


    return pd.DataFrame([{
        "routes_all_tasks": number_of_routes_before_filter,
        "routes_with_task_filter": number_of_routes_after_filter,
        "days_observed": number_of_days_observed,
        "trips_all_tasks": number_of_trips_before_filter,
        "trips_with_task_filter": number_of_trips_after_filter,
        "requests_no_filter": number_requests_no_filter,
        "requests_with_time_filter": number_requests_after_time,
        "requests_with_task_filter": number_requests_after_tasks,
    }])

# ================================================================
# SUMMARIZE ROUTES AND TRIPS
# ================================================================

# ROUTES DRIVEN PER DAY (basic statistics)
def routes_count_by_day(df_keep: pd.DataFrame, save_hist: str | None = None):
    """
    Returns:
      routes_per_day_df: DataFrame[date, routes_count]
      stats: dict with mean, median, percentiles
    Produces (optional) histogram with mean/median lines.
    """
    # Ensure date is datetime
    dates = pd.to_datetime(df_keep["date"], errors="coerce")
    routes_per_day = (
        df_keep.assign(_date=dates)
               .groupby("_date")["route_id"]
               .nunique()
               .sort_index()
    )

    route_stats = pd.DataFrame([{
        "count_days": int(routes_per_day.shape[0]),
        "mean_routes_per_day": float(routes_per_day.mean()),
        "median_routes_per_day": float(routes_per_day.median()),
        "p10": float(routes_per_day.quantile(0.10)),
        "p25": float(routes_per_day.quantile(0.25)),
        "p75": float(routes_per_day.quantile(0.75)),
        "p90": float(routes_per_day.quantile(0.90)),
    }])

    return route_stats


# ROUTES DRIVEN PER WEEKDAY (boxplot)
def routes_count_boxplot_by_weekday(df_keep: pd.DataFrame, save_plots: bool = True, show_plots: bool = True, output_dir: Optional[Path | str] = None, return_df: bool = False):
    """
    Builds a boxplot of routes/day by weekday.
    Returns:
      routes_by_weekday_df: date-level table with [date, weekday, routes_count]
    """
    df = df_keep.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # One row per date: routes_count
    routes_by_day = (
        df.groupby("date")["route_id"].nunique().reset_index(name="routes_count")
    )
    routes_by_day["weekday"] = routes_by_day["date"].dt.day_name()

    # order Mon..Sun (only keep present labels)
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    labels = [d for d in order if d in routes_by_day["weekday"].unique()]

    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_routes_and_trips"
            except Exception:
                outdir = Path("output") / "summarize_routes_and_trips"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "boxplot_routes_per_weekday"

        data = [routes_by_day.loc[routes_by_day["weekday"] == wd, "routes_count"].values
                for wd in labels]
        plt.figure(figsize=(8,5))
        plt.boxplot(data, labels=labels, vert=True)
        plt.title("Routes per weekday (boxplot)")
        plt.ylabel("Routes per day")
        plt.tight_layout()
        plt.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white" )

    if show_plots:
        plt.show()

    return routes_by_day if return_df else None


# VARIABILITY IN NUMBER OF TASKS PER ROUTE

# A route (route_id) is driven on several days. Is the number of tasks assigned to each route stable across the days? So: does a route_id always has more or less the same number of tasks or does a route have one day a lot of tasks and another day much less tasks?
# Task variability: measure used is simple coefficient of variation (cv): standard deviation divided by mean : route_stats["cv"] = route_stats["std"] / route_stats["mean"]; other possibility is to include interquartile range, but would lead us to far.
# Interpretation: how large is the variation relative to the average size of the route?
# Treshold for calling a route "stable" default: <0.25, can be changed in parameters

def tasks_variability_by_route(trip_table: pd.DataFrame, var_treshold: float = 0.25, save_plots: bool = True, show_plots: bool = True, output_dir: Optional[Path | str] = None):
    # Per-route arrays of trip task means
    g = trip_table.groupby("route_id")["mean_tasks_trip"]

    route_stats = g.agg(
        route_days="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
        std="std",
    ).reset_index()
    route_stats["range"] = route_stats["max"] - route_stats["min"]
    route_stats["cv"] = route_stats["std"] / route_stats["mean"]  # coefficient of variation

    # Stability label
    def label(cv):
        if pd.isna(cv): return "only_one_trip"
        if cv < var_treshold: return "stable"
        return "variable"
    route_stats["stability"] = route_stats["cv"].apply(label)

    df = route_stats.copy()

    order = ["stable", "variable", "only_one_trip"]
    labels_present = [c for c in order if c in df["stability"].unique()]

    counts = df["stability"].value_counts().reindex(labels_present, fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100).round(1)

    top15_range_task_var_routes = (
        route_stats.sort_values("range", ascending=False)
                    .head(15)
                    [["route_id", "min", "max", "range"]])

    df_task_variability_by_route = (
        pd.DataFrame({"category": labels_present, "count": counts.values, "pct": pct.values})
    )

    # Plot simple counts bar
    colors = {
        "stable":      "#1f77b4",
        "variable":    "#ff7f0e",
        "only_one_trip": "#919496"

    }
    bar_colors = [colors[c] for c in labels_present]

    plt.figure(figsize=(8, 5))
    plt.bar(labels_present, counts.values, color=bar_colors, edgecolor="white")

    for i, v in enumerate(counts.values):
        plt.text(i, v + max(1, total*0.015), f"{v} ({pct.values[i]:.1f}%)", ha="center")

    plt.title(f"Variability in tasks for {total} routes, treshold={var_treshold})")
    plt.ylabel("Number of routes")
    plt.xlabel("Category")
    plt.tight_layout()

    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_routes_and_trips"
            except Exception:
                outdir = Path("output") / "summarize_routes_and_trips"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "task_variability_per_route.png"
        plt.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white" )

    if show_plots:
        plt.show()

    return route_stats, df_task_variability_by_route, top15_range_task_var_routes


# VARIABILITY IN NUMBER OF TASKS WITHIN TRIPS

# The planner makes several requests per trip (route+day). The number of tasks in a trip can change during that planning phase.
# How often does the number of tasks change during the planning of a trip? How big is the differcence?

def task_variability_within_trip(df_keep: pd.DataFrame, save_plots: bool = True, show_plots: bool = True, output_dir: Optional[Path | str] = None):
    if df_keep.empty:
        return pd.DataFrame()

    # Ensure correct types
    df2 = df_keep.copy()
    df2["num_tasks"] = pd.to_numeric(df2["num_tasks"], errors="coerce")
    df2 = df2.sort_values(["route_id", "date", "request_time"], na_position="last")

    # Pre-calc: first vs last
    first_last = (
        df2.groupby(["route_id", "date"])
        .apply(lambda g: pd.Series({
            "first_tasks": g["num_tasks"].iloc[0],
            "last_tasks":  g["num_tasks"].iloc[-1],
            "delta_last_first": g["num_tasks"].iloc[-1] - g["num_tasks"].iloc[0]
        }))
        .reset_index()
    )

    # Aggregate variability stats
    trip_var = (
        df2.groupby(["route_id", "date"])["num_tasks"]
        .agg(
            min_tasks="min",
            max_tasks="max",
            mean_tasks="mean",
            std_tasks=lambda x: np.std(x, ddof=1) if x.count() > 1 else 0.0,
            requests_in_trip="count",
            nunique_tasks="nunique"
        )
        .reset_index()
    )

    trip_var["range_tasks"] = trip_var["max_tasks"] - trip_var["min_tasks"]
    trip_var["changed_within_trip"] = trip_var["nunique_tasks"] > 1
    trip_var["cv_tasks"] = trip_var["std_tasks"] / trip_var["mean_tasks"]

    # Merge first/last
    trip_var = trip_var.merge(first_last, on=["route_id", "date"], how="left")

    # Summarize trip variability statistics
    summary_trip_var = {
        "total_trips": trip_var.shape[0],
        "trips_with_changes": int(trip_var["changed_within_trip"].sum()),
        "pct_changed": float(trip_var["changed_within_trip"].mean() * 100),
        "cv_mean": trip_var["cv_tasks"].mean(),
        "cv_median": trip_var["cv_tasks"].median(),
        "cv_p90": trip_var["cv_tasks"].quantile(0.90),
        "cv_p95": trip_var["cv_tasks"].quantile(0.95),
    }
    summary_trip_var_df = pd.DataFrame([summary_trip_var])


    top15_range_task_var_trip = (
        trip_var.sort_values("range_tasks", ascending=False)
                .head(15)
                [["route_id", "date", "min_tasks", "max_tasks", "range_tasks"]]
    )

    # Plot range (difference between max and min tasks within a trip) (distribution)
    fig, ax = plt.subplots(figsize=(6,4), dpi=150)
    ax.hist(trip_var["range_tasks"], bins=50, color="#118D57", edgecolor="white")
    ax.set_title("Range of Task Count Within Trips")
    ax.set_xlabel("Max - Min Tasks (range)")
    ax.set_ylabel("Number of Trips")

    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_routes_and_trips"
            except Exception:
                outdir = Path("output") / "summarize_routes_and_trips"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "distribution_of_range_tasks_within_trip.png"

        fig.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white" )

    if show_plots:
        plt.show()

    return trip_var, summary_trip_var_df, top15_range_task_var_trip


# ================================================================
# SUMMARIZE REQUESTS
# ================================================================
def summarize_requests(
    trip_table: pd.DataFrame,
    save_plots: bool = True,
    show_plots: bool = True,
    output_dir: Optional[Path | str] = None
):

    df = trip_table.copy()

    # Ensure types
    df["requests_count"] = pd.to_numeric(df["requests_count"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Derive weekday if missing
    if "weekday" not in df.columns:
        df["weekday"] = df["date"].dt.day_name()

    # REQUESTS PER ROUTE (ROUTE_ID)
    # Number of requests per route (route_id)
    route_summary = (
        df.groupby("route_id")["requests_count"]
          .agg(mean_requests_per_route="mean",
               median_requests_per_route="median",
               days_observed="count")
          .reset_index()
          .sort_values("mean_requests_per_route", ascending=False)
    )

    # Routes with the highest mean number of requests across all days observed
    top15_requests_routes = route_summary.head(15).copy()[["route_id", "mean_requests_per_route"]]

    # REQUESTS PER TRIP (ROUTE/DAY)
    # Number of requests per trip (route/day)
    mean_requests_per_trip = df["requests_count"].mean()
    median_requests_per_trip = df["requests_count"].median()
    p10 = df["requests_count"].quantile(0.10)
    p25 = df["requests_count"].quantile(0.25)
    p75 = df["requests_count"].quantile(0.75)
    p90 = df["requests_count"].quantile(0.90)
    iqr = p75 - p25

    distribution_stats = pd.DataFrame([{
        "count_trips": int(df.shape[0]),
        "mean": float(mean_requests_per_trip),
        "median": float(median_requests_per_trip),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "iqr": float(iqr),
    }])

    top15_requests_trips = df.sort_values("requests_count", ascending=False).head(15).copy()[["route_id", "date", "requests_count"]]

    # Plot: distribution of requests trips
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "distribution_of_requests_per_trip.png"

        plt.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white" )
        plt.figure(figsize=(8,5))
        plt.hist(df["requests_count"].dropna().values, bins=20,
                    color="#4C78A8", edgecolor="white")
        plt.axvline(mean_requests_per_trip, color="red", linestyle="--", label=f"mean = {mean_requests_per_trip:.1f}")
        plt.axvline(median_requests_per_trip, color="green", linestyle=":", label=f"median = {median_requests_per_trip:.1f}")
        plt.title("Requests per trip (distributie)")
        plt.xlabel("Requests per trip")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    # Plot: number of requests per weekday
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "boxplot_of_requests_per_weekday.png"
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        labels = [wd for wd in order if wd in df["weekday"].unique()]
        data = [df.loc[df["weekday"] == wd, "requests_count"].dropna().values for wd in labels]
        plt.figure(figsize=(8,5))
        plt.boxplot(data, labels=labels, vert=True)
        plt.title("Requests per weekday (boxplot)")
        plt.ylabel("Requests per trip")
        plt.tight_layout()
        plt.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

    if show_plots:
        plt.show()

    return route_summary, distribution_stats, top15_requests_routes, top15_requests_trips

# # ============================================================
# # SUMMARIZE TYPE OF REQUESTS (config_name)
# # ============================================================
def summarize_type_of_requests(
    df_keep: pd.DataFrame,
    trip_table: pd.DataFrame,
    save_plots: bool = True,
    show_plots: bool = True,
    output_dir: Optional[Path | str] = None,
):

   # 1) OVERALL config mix (fleet-level, request counts)
    cfg_counts = df_keep["config_name"].value_counts(dropna=False)
    cfg_pct = (cfg_counts / cfg_counts.sum() * 100.0).round(2)

    # Dynamic threshold based on overall EstimateTime share
    estimate_threshold_pct = float(cfg_pct.get("EstimateTime", 0.0))

    # PER-ROUTE type op requests
    per_route = (
        trip_table.groupby("route_id")[["EstimateTime", "CreateSequence", "AddToSequence"]]
                  .sum()
                  .rename_axis("route_id")
    )

    per_route["total"] = per_route.sum(axis=1).replace(0, np.nan)
    per_route_pct = per_route.div(per_route["total"], axis=0) * 100.0
    per_route_pct = per_route_pct.rename(columns={
        "EstimateTime": "estimate_pct",
        "CreateSequence": "create_pct",
        "AddToSequence": "add_pct"
    })
    per_route_pct = per_route_pct.reset_index()

    # Routes above threshold
    routes_over_threshold = (
        per_route_pct.loc[per_route_pct["estimate_pct"] > estimate_threshold_pct]
        .sort_values("estimate_pct", ascending=False)
        .reset_index()
    )

    top15_routes_estimate = routes_over_threshold.head(15)[["route_id", "estimate_pct"]]

    # PLOT overall mix
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        saved_path = outdir / "bar_plot_type_of_requests.png"

        order = ["CreateSequence", "EstimateTime", "AddToSequence"]
        vals = [cfg_pct.get(k, 0.0) for k in order]
        colors = ["#4C78A8", "#F58518", "#54A24B"]

        plt.figure(figsize=(7, 5))
        plt.bar(order, vals, color=colors)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha="center")

        plt.title("Type of requests (filtered on time and tasks)")
        plt.ylabel("% of requests")
        plt.tight_layout()
        plt.savefig(saved_path, dpi=200, bbox_inches="tight", facecolor="white")

        if show_plots:
            plt.show()

    # BOX PLOT: EstimateTime requests per weekday
    if save_plots:
        if output_dir is None:
            try:
                from learning_driver_preferences.paths import OUTPUT
                outdir = Path(OUTPUT) / "summarize_requests"
            except Exception:
                outdir = Path("output") / "summarize_requests"
        else:
            outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Ensure weekday exists
        tt = trip_table.copy()
        if "weekday" not in tt.columns:
            if "date" in tt.columns:
                tt["date"] = pd.to_datetime(tt["date"], errors="coerce")
                tt["weekday"] = tt["date"].dt.day_name()
            else:
                raise ValueError("trip_table must include 'weekday' or 'date' to derive 'weekday'.")

        # Safety: make sure EstimateTime is numeric (counts)
        tt["EstimateTime"] = pd.to_numeric(tt["EstimateTime"], errors="coerce").fillna(0)

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        labels = [wd for wd in order if wd in tt["weekday"].unique()]

        # include zeros (distribution across all trips)
        saved_path_all = outdir / "boxplot_estimatetime_per_weekday_all_trips.png"
        data_all = [tt.loc[tt["weekday"] == wd, "EstimateTime"].values for wd in labels]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data_all, labels=labels, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#F58518", alpha=0.5))
        plt.title("EstimateTime requests per weekday (all trips)")
        plt.ylabel("EstimateTime-requests per trip")
        plt.tight_layout()
        plt.savefig(saved_path_all, dpi=200, bbox_inches="tight", facecolor="white")
        if show_plots:
            plt.show()

    return per_route_pct, top15_routes_estimate


# # ============================================================
# # (4) RELATIONSHIPS (#tasks vs #requests; #tasks vs Estimate/Create shares)
# # ============================================================
def relationship_analysis(trip_table: pd.DataFrame):
    # Correlate workload and behavior
    # Returns Pearson and Spearman correlations, plus simple slopes (np.polyfit).

    out = {}

    def _corr_and_slope(x: pd.Series, y: pd.Series) -> Dict[str, float]:
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        m = pd.DataFrame({"x": x, "y": y}).dropna()
        if m.shape[0] < 3:
            return {"pearson": np.nan, "spearman": np.nan, "slope": np.nan, "n": m.shape[0]}
        pear = m["x"].corr(m["y"], method="pearson")
        spear = m["x"].corr(m["y"], method="spearman")
        slope = float(np.polyfit(m["x"], m["y"], 1)[0])  # y ~ a*x + b, return a
        return {"pearson": float(pear), "spearman": float(spear), "slope": slope, "n": int(m.shape[0])}

    # A) Are more tasks -> more requests?
    out["tasks_vs_requests"] = _corr_and_slope(trip_table["mean_tasks_trip"], trip_table["requests_count"])

    # B) Are more tasks -> higher EstimateTime share?
    out["tasks_vs_estimate_pct"] = _corr_and_slope(trip_table["mean_tasks_trip"], trip_table["estimate_pct"])

    # C) Are more tasks -> higher CreateSequence share?
    out["tasks_vs_create_pct"] = _corr_and_slope(trip_table["mean_tasks_trip"], trip_table["create_pct"])

    # D) (Optional) Are more requests -> higher EstimateTime share?
    out["requests_vs_estimate_pct"] = _corr_and_slope(trip_table["requests_count"], trip_table["estimate_pct"])

    return out
