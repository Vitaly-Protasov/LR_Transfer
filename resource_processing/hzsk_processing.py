import glob
import os
import pathlib
import re
import time

import tqdm
from config import hzsk_dict_match


def determine_lang(file_lines, guess_lang=False):
    curr_lang = None
    lang_pattern = '<languages-used><language lang="'
    # text_pattern = '<event start='
    possible_lang_dict = {}
    for line in file_lines:
        if lang_pattern in line:
            curr_lang = (
                line.split(lang_pattern)[1].split('"/></languages-used>')[0].lower()
            )
            return curr_lang
    if guess_lang and len(possible_lang_dict) > 0:
        print(possible_lang_dict)
        best_match = [
            k
            for k, v in sorted(
                possible_lang_dict.items(), key=lambda item: item[1], reverse=True
            )
        ][0]
        return best_match
    return curr_lang


def hzsk_parsing(hzsk_url: str, path_to_hzsk_folder: str):
    text_pattern = "<event start="
    temp_folder = "test_folder/"
    filename = hzsk_url.split("/")[-1]

    # os.system(f'wget {url}')
    # os.system(f'unzip {filename} -d {temp_folder}')

    exb_paths = glob.glob(f"{temp_folder}**/*.exb", recursive=True)
    for exb_one_path in tqdm(exb_paths):
        extracted_text = []
        current_file_name = pathlib.Path(exb_one_path).stem
        f_old_data = list(open(exb_one_path))
        curr_lang = determine_lang(f_old_data)
        # if curr_lang == None:
        #     for line in f_old_data:
        #         if text_pattern in line and current_file_name not in line:
        #             try:
        #                 _text = re.match(r'<event start=".*" end=".*">(.*)</event>', line).group(1)
        #                 extracted_text.append(_text.strip())
        #             except:
        #                 pass
        if curr_lang != None:
            new_folder_path = pathlib.Path(
                path_to_hzsk_folder, hzsk_dict_match[curr_lang]
            )
            if not os.path.exists(new_folder_path):
                os.mkdir(new_folder_path)
                time.sleep(0.2)

            new_file_path = pathlib.Path(
                new_folder_path, pathlib.Path(exb_one_path).stem + ".txt"
            )

            if len(extracted_text):
                with open(new_file_path, "w") as f_new:
                    for i in extracted_text:
                        f_new.write(i + "\n")
        else:
            raise "No language was detected"

    # shutil.rmtree(temp_folder, ignore_errors=True)
    # os.remove(filename)
