import os
from pathlib import PurePath

import pandas as pd
from utils import extract_url_dataframe

if __name__ == "__main__":
    # prepare data
    data_folder = "../data/"
    bib_path = PurePath(data_folder, "anthology+abstracts.bib")
    language_path = PurePath(data_folder, "wals_language.csv")
    if not all([os.path.exists(i) for i in [bib_path, language_path]]):
        os.system("bash load_data.sh")

    df = pd.read_csv(language_path)
    list_langs = df["Name"].to_list()
    df = extract_url_dataframe(bib_path, list_langs)
    df_out_path = PurePath(data_folder, "lang_url.csv")
    df.to_csv(df_out_path, index=None)
