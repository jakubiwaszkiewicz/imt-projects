import pandas as pd

def parsing(data_IO, data_non_IO):
    data_IO_processed = data_IO[["post_text"]].copy()
    data_IO_processed["is_russian_disinformation"] = True

    data_non_IO_processed = data_non_IO[["C6"]].copy()
    data_non_IO_processed.rename(columns={"C6": "post_text"}, inplace=True)
    data_non_IO_processed["is_russian_disinformation"] = False

    data_all = pd.concat([data_IO_processed, data_non_IO_processed], axis=0, ignore_index=True)

    return data_all
