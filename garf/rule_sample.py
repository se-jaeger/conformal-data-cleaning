import sqlite3


def rule_sample(path_rules, path, order):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM '{path}'")
    data_all = cursor.fetchall()  # All data
    data_all = [x[:-1] for x in data_all]  # Removing the Label column

    if order == 1:
        print("Positive sequence sampling data to production rules……")

    elif order == 0:
        print("Inverse order sampling data to production rules……")
        data_all = [x[::-1] for x in data_all]

    rule = ""
    for data_tuple in data_all:
        rule_tuple = ""
        for data_cell in data_tuple:
            if data_cell is None:
                continue
            else:
                if rule_tuple == "":
                    rule_tuple = f"{data_cell}"
                else:
                    rule_tuple += f",{data_cell}"
        rule += rule_tuple + "\n"

    with open(path_rules, "w", encoding="utf-8") as f:
        f.write(rule)

    return len(data_all)
