import glob
import os
import pathlib
import time
from typing import List

from config import hand_matched_dict
from tqdm import tqdm


def extract_text_from_conllu(old_file_path: str, new_file_path: str):
    pattern = "# text = "
    f_old = open(old_file_path, "r")
    f_new = open(new_file_path, "w")
    for line in f_old:
        if line.startswith(pattern):
            f_new.write(line[len(pattern) :])


def ud_processing(
    path_to_ud_folder: str, path_to_save_folder: str, langs_names_list: List[str]
):
    good_langs_list = []
    for foldername in tqdm(os.listdir(path_to_ud_folder)):
        lang_name = foldername[3:].split("-")[0].replace("_", "-")
        if lang_name in langs_names_list:
            good_langs_list.append(lang_name)
            new_folder_path = pathlib.Path(path_to_save_folder, lang_name)
        elif lang_name in hand_matched_dict:
            good_langs_list.append(hand_matched_dict[lang_name])
            new_folder_path = pathlib.Path(
                path_to_save_folder, hand_matched_dict[lang_name]
            )
        else:
            continue

        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
            time.sleep(0.2)

        each_lang_path = pathlib.Path(path_to_ud_folder, foldername)
        conllu_files = glob.glob(f"{each_lang_path}/*.conllu")
        for file_path in conllu_files:
            new_file_path = pathlib.Path(
                new_folder_path, pathlib.Path(file_path).stem + ".txt"
            )
            extract_text_from_conllu(file_path, new_file_path)
