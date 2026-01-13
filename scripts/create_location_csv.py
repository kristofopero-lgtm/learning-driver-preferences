import os
import json
import csv

def create_location_csv():
    """
    This script processes JSON files from the 'data/requests' directory,
    extracts latitude and longitude data, and compiles it into a single
    CSV file named 'locations.csv' within the 'data' directory.

    The script iterates through each subdirectory in 'data/requests', treating
    each as a unique group identified by the directory name. For each JSON
    file found, it extracts the 'latitude' and 'longitude' from the 'gnss'
    object.

    The resulting CSV file will have three columns: 'latitude', 'longitude',
    and 'groupedid', where 'groupedid' helps in tracing the data back to
    its original folder.
    """
    requests_dir = os.path.join('data', 'requests')
    output_file = os.path.join('data', 'locations.csv')

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['latitude', 'longitude', 'groupedid']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for group_folder in os.listdir(requests_dir):
            group_path = os.path.join(requests_dir, group_folder)
            if os.path.isdir(group_path):
                for filename in os.listdir(group_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(group_path, filename)
                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                if 'tasks' in data and isinstance(data['tasks'], list):
                                    for task in data['tasks']:
                                        if 'address' in task and 'latitude' in task['address'] and 'longitude' in task['address']:
                                            writer.writerow({
                                                'latitude': task['address']['latitude'],
                                                'longitude': task['address']['longitude'],
                                                'groupedid': group_folder
                                            })
                            except json.JSONDecodeError:
                                print(f"Error decoding JSON from file: {file_path}")

if __name__ == "__main__":
    create_location_csv()

