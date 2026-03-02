from glob import glob
import pandas as pd

class Loader:
    def __init__(self, is_one_file, small_IO_data_regex, all_IO_data_regex):
        self.is_one_file = is_one_file
        self.small_IO_data_regex = small_IO_data_regex
        self.all_IO_data_regex = all_IO_data_regex
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.expand_frame_repr", False)

    def _files_strategy(self):
        if self.is_one_file:
            all_IO_files = glob(self.small_IO_data_regex)
        else:
            all_IO_files = glob(self.all_IO_data_regex)
        return all_IO_files

    def IO_data(self):
        all_IO_files = self._files_strategy()
        data_list = []
        for file in all_IO_files:
            parquet = pd.read_parquet(file)
            data_list.append(parquet)
        data = pd.concat(data_list, axis=0, ignore_index=True)
        return data

    def non_IO_data(self):
        non_IO_data_regex = "./data_preprocessing/data/01_raw/not_information_operation/*.csv"
        files = glob(non_IO_data_regex)
        if not files:
            raise FileNotFoundError()

        column_names = ["C1", "C2", "C3", "C4", "C5", "C6"]  # <-- zmień na swoje nazwy
        data_list = [pd.read_csv(file, encoding='latin1', header=None, names=column_names) for file in files]

        data = pd.concat(data_list, axis=0, ignore_index=True)
        return data
