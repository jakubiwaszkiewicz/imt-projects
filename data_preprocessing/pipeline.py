from .nodes.loader import Loader
from .nodes.parsing import parsing
from .nodes.vectorizer import vectorizer

def data_preprocessing_pipeline():

    small_IO_data_regex = "./data_preprocessing/data/01_raw/information_operation/Russia_1/*"
    all_IO_data_regex = "./data_preprocessing/data/01_raw/information_operation/Russia_*/*"

    loader = Loader(
        is_one_file=False,
        small_IO_data_regex = small_IO_data_regex,
        all_IO_data_regex = all_IO_data_regex
    )


    data_IO = loader.IO_data()
    data_non_IO = loader.non_IO_data()


    data = parsing(data_IO, data_non_IO)

    X, X_texts, y = vectorizer(data)


    return X, X_texts, y