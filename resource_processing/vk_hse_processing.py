import glob
import json
import os
import pathlib
from typing import List

import pandas as pd
import tqdm
from config import list_archives, list_archives2


def load_web(list_files: List[str] = list_archives):
    for i in tqdm.tqdm(list_files):
        os.system(
            f"wget http://web-corpora.net/wsgi3/minorlangs/static/files/web_corpus/{i}"
        )
        os.system(f"unzip {i}")
        os.system(f"rm -r {i}")


def load_vk(list_files: List[str] = list_archives2):
    for i in tqdm.tqdm(list_files):
        os.system(
            f"wget http://web-corpora.net/wsgi3/minorlangs/static/files/vk_corpus/{i}"
        )
        os.system(f"unzip {i} -d corpora/")
        os.system(f"rm -r {i}")


def process_vk_hse(save_folder_path: str, main_folder_path: str, df_wals: pd.DataFrame):
    iso_to_lang_dict = dict(zip(df_wals.iso_code.to_list(), df_wals.Name.to_list()))
    iso_to_lang_dict["bxr"] = "Buriat"

    text_pattern = '"text":'
    for i in tqdm.tqdm(os.listdir(main_folder_path)):
        lang_code = i.split("_")[0]
        save_lang_folder = pathlib.Path(save_folder_path, iso_to_lang_dict[lang_code])
        if not os.path.exists(save_lang_folder):
            os.mkdir(save_lang_folder)

        for file in glob.glob(
            f"{pathlib.Path(main_folder_path, i)}**/*.json", recursive=True
        ):
            file_path_old = pathlib.Path(file)
            file_path_new = pathlib.Path(save_lang_folder, pathlib.Path(file).stem)
            f_old = open(file_path_old)
            f_new = open(file_path_new, "w")
            for line in list(f_old):
                if text_pattern in line:
                    extracted_text = line.split(text_pattern)[-1].strip()[1:-2]
                    if extracted_text:
                        f_new.write(extracted_text + "\n")
