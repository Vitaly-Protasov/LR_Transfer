import os
import pathlib
from typing import List, Tuple

from cchardet import detect
from tqdm import tqdm_notebook
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast


def detect_encoding_of_file(filename: str):
    with open(filename, "rb") as target_file:
        return detect_encoding_of_data(target_file.read())


def detect_encoding_of_data(data: bytes):
    return detect(data)["encoding"]


def read_text_with_autodetected_encoding(filename: str):
    with open(filename, "rb") as target_file:
        data = target_file.read()

    if not data:
        return ""  # In case of empty file, return empty string

    encoding = detect_encoding_of_data(data) or "utf-8"
    return data.decode(encoding)


def count_tokens_symbols_in_file(
    path_to_file: str, tokenizer: T5TokenizerFast
) -> Tuple[int, int]:
    num_tokens = 0
    num_symbols = 0
    f = open(path_to_file)
    for line in list(f):
        num_tokens += len(tokenizer(line)["input_ids"])
        num_symbols += len(line.replace(" ", ""))
    return num_tokens, num_symbols


def get_num_tokens_symbols_language(
    languages: List[str], tokenizer: T5TokenizerFast, path_to_folder_with_languages: str
) -> List[str]:
    formed_statistics = []
    set_existed_langs = set(os.listdir(path_to_folder_with_languages))
    for lang_name in tqdm_notebook(languages):
        lang_tokens = 0
        lang_symbols = 0
        if lang_name in set_existed_langs:
            p1 = pathlib.Path(path_to_folder_with_languages, lang_name)
            for file_name in tqdm_notebook(os.listdir(p2)):
                p = pathlib.Path(p1, file_name)
                try:
                    num_tokens, num_symbols = count_tokens_symbols_in_file(p, tokenizer)
                    lang_tokens += num_tokens
                    lang_symbols += num_symbols
                except:
                    print(f"Error in language {lang_name}. Path: {p}")

            formed_statistics.append(f"{lang_name}|{lang_tokens}, {lang_symbols}")
    return formed_statistics
