import sqlite3
from typing import Optional

import pandas as pd


def create_tables_for_dataset(connection: sqlite3.Connection, name: str, path: str, path_dirty: str):
    df_clean = pd.read_csv(path, dtype=str)

    df_clean["Label"] = None
    df_clean.to_sql(f"{name}", connection, if_exists="replace", index=False)

    df_dirty = pd.read_csv(path_dirty, dtype=str)

    df_dirty["Label"] = None
    df_dirty.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
    df_dirty.to_sql(f"{name}_dirty", connection, if_exists="replace", index=False)


def create_database(datasets: list[dict], database_path: Optional[str] = None):
    if database_path is None:
        connection = sqlite3.connect("database.db")
    else:
        connection = sqlite3.connect(database_path)

    for d in datasets:
        print(f"Creating tables for {d['name']} from clean file at {d['path']}, dirty file at {d['path_dirty']}.")
        create_tables_for_dataset(connection, d["name"], d["path"], d["path_dirty"])

    connection.close()
    print("Database successfully created.")


if __name__ == "__main__":
    # datasets = {"Hosp_rules": "./data/Hosp_clean.csv", "Food": "./data/Food_clean.csv"}
    datasets = [
        {
            "name": "40498",
            "path": "./data/test/40498_X.csv",
            "path_dirty": "./data/corrupted/GaussianNoise/0.3/40498_X.csv",
        },
    ]

    create_database(datasets)
