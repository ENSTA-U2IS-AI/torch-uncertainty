import csv


def csv_writter(path, dic):
    # Check if the file already exists
    if path.is_file():
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"

    # Write dic
    with open(path, rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(dic.keys())
        writer.writerow([f"{elem:.4f}" for elem in dic.values()])
