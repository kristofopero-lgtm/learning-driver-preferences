import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

def group_locations(input_file='data/locations.csv', output_file='data/locations_grouped.csv'):
    """
    Groups locations from a CSV file based on their proximity using DBSCAN.

    This script reads a CSV file containing latitude and longitude coordinates,
    and groups them using the DBSCAN clustering algorithm, which is highly
    optimized for this task.

    A distance threshold of 10 meters is used.
    """
    # 10 meters in degrees
    eps_meters = 10
    eps_degrees = eps_meters / 111000.0

    print("Reading data...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    # Extract coordinates for clustering
    coords = df[['latitude', 'longitude']].values

    print("Clustering locations with DBSCAN...")
    db = DBSCAN(eps=eps_degrees, min_samples=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1).fit(coords)
    
    # The 'n_jobs=-1' parameter uses all available CPU cores for parallel processing.

    df['close_group'] = db.labels_

    num_groups = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Found {num_groups} groups.")

    print(f"Saving grouped locations to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    group_locations()