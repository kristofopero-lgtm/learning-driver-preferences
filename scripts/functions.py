import json
import os
import pandas as pd


def create_df_of_requests(path_to_parent_folder):
    dirs = os.listdir(path_to_parent_folder)
    if "README.txt" in dirs:
        dirs.pop(dirs.index("README.txt")) # Remove README.txt folder

    rows = []

    for dir in dirs:
        path = os.path.join(path_to_parent_folder, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                route_id = file.split('-')[0].split('_')[1]
                date = pd.to_datetime(file.split('-')[1], format="%Y%m%d")
                time = pd.to_datetime(file.split('-')[2], format="%H%M%S").time()
                with open(os.path.join(path, file), "r") as f:                    
                    data = json.load(f)
                    request_type = data["configurationName"]
                    request_id = data["id"]
                    tasks = data["tasks"]                    
                    fixed_tasks = [task["taskId"] for task in data["fixedTasks"]]
                    for task in tasks:
                        row = {}
                        row_id = file.split('.')[-2] + '-' + task["id"]
                        row["row_id"] = row_id
                        row["route_id"] = route_id
                        row["date"] = date
                        row["time"] = time
                        row["request_id"] = request_id
                        row["request_type"] = request_type
                        row["task_id"] = task["id"]
                        loc = task["address"]
                        row["lat"] = loc["latitude"]
                        row["lon"] = loc["longitude"]
                        row["fixed"] = True if row["task_id"] in fixed_tasks else False
                        row["position_fixed"] = (
                            fixed_tasks.index(row["task_id"]) if row["fixed"] else None
                        )
                        rows.append(row)

    return pd.DataFrame(rows)


def create_df_of_responses(path_to_parent_folder):
    dirs = os.listdir(path_to_parent_folder)

    rows = []

    for dir in dirs:
        path = os.path.join(path_to_parent_folder, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                route_id = file.split('-')[0].split('_')[1]
                date = pd.to_datetime(file.split('-')[1], format="%Y%m%d")
                time = pd.to_datetime(file.split('-')[2], format="%H%M%S").time()
                with open(os.path.join(path, file), "r") as f:
                       tasks = f.readlines()
                       for task in tasks:
                        row = {}
                        row_id = file.split('.')[-2] + '-' + task.strip()
                        row["row_id"] = row_id
                        row['route_id'] = route_id
                        row['date'] = date
                        row['time'] = time                        
                        row['task_id'] = task.strip()
                        row['task_sequence_number'] = tasks.index(task) + 1
                        rows.append(row)

    return pd.DataFrame(rows)


def join_requests_and_responses(requests_df, responses_df):
    return requests_df.merge(
        responses_df[["row_id", "task_sequence_number"]],
        on='row_id',
        how="inner"
    )