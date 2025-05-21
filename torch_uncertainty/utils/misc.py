import csv
from pathlib import Path


def csv_writer(path: Path, dic: dict) -> None:
    """Write a dictionary to a csv file.

    Args:
        path (Path): Path to the csv file.
        dic (dict): Dictionary to write.
    """
    if path.is_file():  # Check that the file already exists
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"
    # Write dic
    with path.open(rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(dic.keys())
        writer.writerow([f"{elem:.4f}" for elem in dic.values()])
