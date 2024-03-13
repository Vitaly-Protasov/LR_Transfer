from collections import OrderedDict
from typing import Dict, List

import pandas as pd
from sklearn.cluster import KMeans


def get_encoding_dict(df: pd.DataFrame):
    n = 1
    encoding_dict = {}
    df_cutted_to_experiments = df.fillna(0)
    for i in df_cutted_to_experiments.columns:
        temp_list = df_cutted_to_experiments[i].to_list()
        for j in temp_list:
            if j not in encoding_dict and isinstance(j, str):
                encoding_dict[j] = n
                n += 1
    return encoding_dict


def encode_all_df(df: pd.DataFrame) -> pd.DataFrame:
    df_test = df.copy()
    encoding_dict = get_encoding_dict(df)
    for i in df.columns:
        df_test[i] = df_test[i].apply(
            lambda x: encoding_dict[x] if isinstance(x, str) else x
        )
    return df_test


def add_clusters_column(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)
    clusters = kmeans.predict(df)

    df_with_clusters = df.copy()
    df_with_clusters["Clusters"] = clusters
    return df_with_clusters


def form_dict_by_clusters_sorted(
    df: pd.DataFrame, column_to_sort: str
) -> Dict[int, Dict[str, int]]:
    final_dict_clusters = {}
    for n_cluster in df.Clusters.unique():
        sub_df = df[df.Clusters == n_cluster]
        if sub_df.shape[0] == 1:
            current_family = sub_df.family.item()
            current_genus = sub_df.genus.item()
            keys = df[(df.family == current_family) & (df.genus == current_genus)][
                "Name"
            ]
            values = df[(df.family == current_family) & (df.genus == current_genus)][
                column_to_sort
            ]
        else:
            keys = sub_df["Name"]
            values = sub_df[column_to_sort]
        each_cluster_dict = dict(zip(keys, values))
        sorted_dict = OrderedDict(
            {
                k: v
                for k, v in sorted(
                    each_cluster_dict.items(), key=lambda item: item[1], reverse=True
                )
            }
        )
        final_dict_clusters[n_cluster] = sorted_dict
    return final_dict_clusters


def form_relation_dict_by_langs(
    dict_by_clusters: Dict[int, Dict[str, int]]
) -> Dict[str, List[str]]:
    relation_dict_by_languages = {}
    for cluster_dict in list(dict_by_clusters.values()):
        source_lang = list(cluster_dict)[0]
        lower_langs = list(cluster_dict)[1:]
        relation_dict_by_languages[source_lang] = lower_langs
    return relation_dict_by_languages
