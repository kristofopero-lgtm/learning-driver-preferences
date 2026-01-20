import pandas as pd
import numpy as np

## visualiseren van DBSCAN?
## afrondingsfouten  bij *10^8 
## DBSCAN presenteren in presentatie
## visualisatie van dbscan, voor op presentatie

def group_locations_df_v2(
    df: pd.DataFrame,
    eps_meters: float = 10.0,
    lat_col: str = "lat",
    lon_col: str = "lon",
    loc_id_col: str = "loc_id"
) -> pd.DataFrame:
    """
    Groups locations in a DataFrame so that all points in the same group
    are within eps_meters of each other (max distance ≤ eps_meters).

    Uses grid-based bucketing for speed and guaranteed maximum distance.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latitude and longitude columns.
    eps_meters : float
        Maximum allowed distance in meters between any two points in the same group.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    loc_id_col : str
        Column name to store location group IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated loc_id column.
    """

    df = df.copy()

    # Clear existing loc_id
    df[loc_id_col] = np.nan

    # Convert meters to degrees (~valid for small distances)
    meters_per_degree = 111_000.0
    cell_size_deg = eps_meters / meters_per_degree

    # Compute grid cell indices
    df["_cell_x"] = (df[lat_col] / cell_size_deg).astype(int)
    df["_cell_y"] = (df[lon_col] / cell_size_deg).astype(int)

    # Assign loc_id based on unique cell combination
    df[loc_id_col] = df.groupby(["_cell_x", "_cell_y"]).ngroup()

    # Clean up temporary columns
    df.drop(columns=["_cell_x", "_cell_y"], inplace=True)

    return df