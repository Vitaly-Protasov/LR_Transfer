import glob
import os
import pathlib

import pandas as pd
from config import ISO_639_1_TO_3
from tqdm import tqdm


def process_wiki_data(wiki_data_folder: str, result_folder: str, df_wals: pd.DataFrame):
    iso_to_lang_dict = dict(zip(df_wals.iso_code.to_list(), df_wals.Name.to_list()))
    iso_to_lang_dict["swa"] = "Swahili"
    iso_to_lang_dict["uzb"] = "Uzbek"
    iso_to_lang_dict["aze"] = "Azerbaijani"
    iso_to_lang_dict["zho"] = "Chinese"
    iso_to_lang_dict["msa"] = "Malay"
    iso_to_lang_dict["mon"] = "Mongolian"
    iso_to_lang_dict["fas"] = "Persian"
    iso_to_lang_dict["bxr"] = "Buriat"
    iso_to_lang_dict["ara"] = "Arabic"

    wiki_files = glob.glob(f"{wiki_data_folder}**/*wiki_*", recursive=True)
    for path in tqdm(wiki_files):
        if not path.endswith("wiki_multilingual"):
            code_lang_1 = pathlib.Path(path).parts[1]
            if code_lang_1 not in iso_to_lang_dict:
                if code_lang_1 not in ISO_639_1_TO_3:
                    print(code_lang_1)
                    continue

                else:
                    code_lang_3 = ISO_639_1_TO_3[code_lang_1]
                    if code_lang_3 in iso_to_lang_dict:
                        lang_name = iso_to_lang_dict[code_lang_3]
                    elif (
                        code_lang_3 in ISO_639_1_TO_3
                        and ISO_639_1_TO_3[code_lang_3] in iso_to_lang_dict
                    ):
                        lang_name = iso_to_lang_dict[ISO_639_1_TO_3[code_lang_3]]
                    else:
                        print("Error: " + code_lang_3)
            else:
                lang_name = iso_to_lang_dict[code_lang_1]

            folder_path = pathlib.Path(result_folder, lang_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            destination_file_path = pathlib.Path(
                folder_path, "_".join(path.split("/")[-2:])
            )
            if " " in lang_name:
                os.system(f"cp -r {path} '{destination_file_path}'")
            else:
                os.system(f"cp -r {path} {destination_file_path}")
