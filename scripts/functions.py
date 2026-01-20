import json
import os
import pandas as pd


def create_df_of_requests(path_to_parent_folder):
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
                    data = json.load(f)
                    request_type = data["configurationName"]
                    request_id = data["id"]
                    tasks = data["tasks"]
                    fixed_tasks = [task["taskId"] for task in data["fixedTasks"]]
                    for task in tasks:
                        row = {}
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
