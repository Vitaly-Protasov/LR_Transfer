import os
import pathlib
from typing import List

import fasttext
import fire
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

model_path = hf_hub_download(
    repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None
)
MODEL = fasttext.load_model(model_path)

PATH_TO_DATASET = "/home/jovyan/XL_Data/jovyan/protasov/XL_Dataset/"
PATH_TO_LOG_RESULTS = "/home/jovyan/protasov/files_with_detected_langs/"


def get_chosen_files(
    lang_folder: os.PathLike, percent_of_random_files: float = 0.01
) -> List[os.PathLike]:
    files_for_lang = list(pathlib.Path(lang_folder).glob("**/*"))
    choice_files_num = int(np.ceil(len(files_for_lang) * percent_of_random_files))

    chosen_files = np.random.choice(files_for_lang, choice_files_num, replace=False)
    return list(chosen_files)


def main(
    path_to_dataset: os.PathLike = PATH_TO_DATASET,
    percent_of_random_files: float = 0.01,
    num_langs_to_detect: int = 2,
    line_sep: str = "\n",
    sep_for_log: str = " >> ",
    log_filename: os.PathLike = "log.txt",
    path_to_log_results: os.PathLike = PATH_TO_LOG_RESULTS,
) -> None:
    langs_by_folders = list(pathlib.Path(path_to_dataset).glob("*"))

    for lang_folder in tqdm(langs_by_folders, desc="Iteration by languages"):
        lang_name = lang_folder.name

        chosen_files = get_chosen_files(lang_folder, percent_of_random_files)
        os.makedirs(
            pathlib.Path(path_to_log_results, lang_name), exist_ok=True
        )  # create folder for logs storage

        lang_log = []
        print(f"Number of files for lang={lang_name} to process: {len(chosen_files)}")
        for file_path in chosen_files:
            with open(file_path) as f:
                f_data = f.read()
                f_lines = f_data.split(line_sep)

                for line in tqdm(
                    f_lines, desc=f"Iteration by lines for {str(file_path)}"
                ):
                    detection_output = MODEL.predict(line, k=num_langs_to_detect)

                    line_for_log = f"{lang_name}{sep_for_log}{file_path}{sep_for_log}[{line}]{sep_for_log}{str(detection_output)}"
                    lang_log.append(line_for_log)

        log_filepath = pathlib.Path(path_to_log_results, lang_name, log_filename)
        with open(log_filepath, "w") as f_log:
            for log_line in lang_log:
                f_log.write(log_line + line_sep)


if __name__ == "__main__":
    fire.Fire(main)
