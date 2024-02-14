import sqlite3
from pathlib import Path


def att_reverse(path, order, models_base_path):
    # Connecting to the database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(f"select * from '{path}'")
    data1 = cursor.fetchall()  # All data

    data1 = [x[:-1] for x in data1]  # Data with label removed
    if order == 0:
        data1 = [x[::-1] for x in data1]  # Reverse order
        des = list(cursor.description)
        des.reverse()
        del des[0]  # Remove the label from the table header

    else:
        des = list(cursor.description)

    print("att_reverse() description:", [item[0] for item in des])
    t2 = len(data1[0])  # Length of data per row

    att_name = []
    for item in des:
        att_name.append(item[0])

    result = {}
    for i in range(t2):
        result[i] = att_name[i]

    with open(Path(models_base_path / "att_name.txt"), "w") as file:
        file.write(str(result))
