import os
import re
import shutil
from time import sleep
from typing import List

import pandas as pd
import tldextract
from IPython.display import clear_output
from tika import parser
from tqdm import tqdm, trange


def extract_indices(file_data: List[str]) -> List[int]:
    articles_index = []
    for i, string in enumerate(file_data):
        if string.startswith("@"):
            articles_index.append(i)
    return articles_index


def filter_language(lang: str) -> str:
    clear_lang = lang.split("(")[0].lower().strip()
    popular_words = ["even"]
    if len(clear_lang) < 4 or clear_lang in popular_words:
        return f"{clear_lang} language"
    return clear_lang


def check_phrase_in_text(phrase, text):
    len_s = len(phrase)
    return any(phrase == text[i : len_s + i] for i in range(len(text) - len_s + 1))


def extract_url_dataframe(bib_path: str, list_langs: List[str]) -> pd.DataFrame:
    f = open(bib_path)
    file_array = list(f)
    articles_index = extract_indices(file_array)

    final_df = {"language": [], "url": []}
    for i in trange(len(articles_index) - 1):
        snippet_array = file_array[articles_index[i] : articles_index[i + 1]]
        for string in snippet_array:
            cleaned_string = string.strip()

            if cleaned_string.startswith("title"):
                temp_title = re.findall(
                    '(?<=").*?(?=")|(?<=\{).*?(?=\})', cleaned_string
                )[0]
                cleared_title = (
                    re.sub("[^a-zA-Z ]+", "", temp_title).lower().strip().split()
                )

            if cleaned_string.startswith("url"):
                temp_url = re.findall('(?<=").*?(?=")', cleaned_string)[0]

            if cleaned_string.startswith("abstract"):
                temp_abstract = re.findall(
                    '(?<=").*?(?=")|(?<=\{).*?(?=\})', cleaned_string
                )[0]
                cleared_abstract = (
                    re.sub("[^a-zA-Z ]+", "", temp_abstract).lower().strip().split()
                )
                for language_name in list_langs:
                    # language_subwords = re.sub('[^a-zA-Z ]+', '', language_name).lower().strip().split()
                    # if all([lang_word in cleared_abstract for lang_word in language_subwords]) or \
                    #     all([lang_word in cleared_title for lang_word in language_subwords]):
                    #     final_df['language'].append(language_name)
                    #     final_df['url'].append(temp_url)
                    language_name_clear = filter_language(language_name).split()
                    abstract_match = check_phrase_in_text(
                        language_name_clear, cleared_abstract
                    )
                    title_match = check_phrase_in_text(
                        language_name_clear, cleared_title
                    )
                    if abstract_match or title_match:
                        final_df["language"].append(language_name)
                        final_df["url"].append(temp_url)

    return pd.DataFrame(final_df)


def get_text_from_pdf(filepath: str) -> str:
    raw = parser.from_file(filepath)
    return raw["content"]


def get_links_from_text(text_of_file: str) -> List[str]:
    array_1 = [url[1:] for url in re.findall("[^//]www\.\S+", text_of_file)]
    array_2 = re.findall("(https://www\.\S+|https://\S+)", text_of_file)
    return array_1 + array_2


def extract_links(data_urls: List[str]) -> pd.DataFrame:
    test_dir = "test_dir/"
    ending = ".pdf"
    final_df_dict = {"article": [], "links": []}

    if not os.getcwd() == test_dir:
        os.mkdir(test_dir)
        os.chdir(test_dir)

    for url in tqdm(data_urls):
        if not url.endswith(ending):
            clear_output(wait=True)
            os.system(f"wget {url}")
            # sleep(0.1)
            try:
                text = get_text_from_pdf(url + ending)
            except:
                continue
        else:
            try:
                text = get_text_from_pdf(url)
            except:
                continue

        links_in_text = get_links_from_text(text)
        unique_links = list(set(links_in_text))
        final_df_dict["article"].extend([url] * len(unique_links))
        final_df_dict["links"].extend(unique_links)

    os.chdir("../")
    shutil.rmtree(test_dir)
    return pd.DataFrame(final_df_dict)


def get_domains(df: pd.DataFrame) -> pd.DataFrame:
    domains = []
    for i, row in df.iterrows():
        domain = tldextract.extract(row.links).domain
        domains.append(domain)
    df["domains"] = domains
    return df
