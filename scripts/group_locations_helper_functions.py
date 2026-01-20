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
