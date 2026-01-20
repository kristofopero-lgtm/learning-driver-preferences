import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def group_locations_df(
    df: pd.DataFrame,
    eps_meters: float = 10.0
) -> pd.DataFrame:
    """
    Groups locations in a DataFrame based on proximity using DBSCAN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing:
        - 'lat' (latitude)
        - 'lon' (longitude)
        - 'loc_id' (existing column to store group IDs)
    eps_meters : float, optional
        Maximum distance (in meters) between points to be considered
        in the same group. Default is 10 meters.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'loc_id' values representing proximity groups.
    """

    # Convert meters to degrees (approximation valid for small distances)
    eps_degrees = eps_meters / 111_000.0

     # Work on a copy to avoid side effects
    df = df.copy()

    # Clear existing loc_id values
    df['loc_id'] = np.nan

    # Extract coordinates
    coords = df[['lat', 'lon']].to_numpy()

    # Run DBSCAN clustering
    db = DBSCAN(
        eps=eps_degrees,
        min_samples=1,
        algorithm='kd_tree',
        metric='euclidean',
        n_jobs=-1
    ).fit(coords)

    # Assign cluster labels to existing loc_id column
    df = df.copy()
    df['loc_id'] = db.labels_

    return df



def add_max_distance_per_loc_id(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    loc_id_col: str = "loc_id",
    output_col: str = "max_group_distance_m"
) -> pd.DataFrame:
    """
    Adds a column containing the maximum distance (in meters) between
    any two points sharing the same loc_id.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing latitude, longitude, and loc_id columns.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    loc_id_col : str
        Location group ID column name.
    output_col : str
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column containing max intra-group distance in meters.
    """

    df = df.copy()

    # Approx conversion: degrees → meters
    METERS_PER_DEGREE = 111_000.0

    max_distances = {}

    for loc_id, group in df.groupby(loc_id_col):
        if len(group) < 2:
            max_distances[loc_id] = 0.0
            continue

        coords = group[[lat_col, lon_col]].to_numpy()

        # Compute bounding box max distance (fast and sufficient)
        lat_range = coords[:, 0].max() - coords[:, 0].min()
        lon_range = coords[:, 1].max() - coords[:, 1].min()

        max_distances[loc_id] = np.sqrt(
            lat_range ** 2 + lon_range ** 2
        ) * METERS_PER_DEGREE

    df[output_col] = df[loc_id_col].map(max_distances)

    return df

