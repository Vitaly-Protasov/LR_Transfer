from pathlib import Path
from tqdm.auto import tqdm
import json
from datetime import datetime
import os
from fire import Fire

from tl_experiments import mt5PerplexityExperiments
from tl_utils import clear_memory


LR_LANGS = [
    "Akan",
    "Atikamekw",
    "Bambara",
    "Bhojpuri",
    "Chichewa",
    "Cantonese",
    "Coptic",
    "Dagbani",
    "Greenlandic (South)",
    "Guaraní",
    "Kashmiri",
    "Kurmanji",
    "Koryak",
    "Komi-Zyrian",
    "Madurese",
    "Nanai",
    "Quiché",
    "Romani (Lovari)",
    "Rundi",
    "Samoan",
    "Sesotho",
    "Shor",
    "Sranan",
    "Swati",
    "Tabassaran",
    "Tat (Muslim)",
    "Tofa",
    "Tsakhur",
    "Tsonga",
    "Udi",
    "Yukaghir (Kolyma)",
]

XL_DATASET_PATH = "/home/jovyan/XL_Data/jovyan/protasov/XL_Dataset/"
ALL_CKPTS_PATH = "/home/jovyan/protasov/LR_Transfer/data/launched_experiments/"
LOG_SAVE_FOLDER = "/home/jovyan/protasov/LR_Transfer/data/logs/"


def main(
    hr_lang: str,
    device: str = "cuda:0",
    model_id: str = "google/mt5-base",
    prefix_training_folder: str = "training_on_",
):
    all_trained_hr_langs = [
        l.split(prefix_training_folder)[-1] for l in os.listdir(ALL_CKPTS_PATH)
    ]
    if hr_lang not in all_trained_hr_langs:
        raise Exception(f"'{hr_lang}' not in '{ALL_CKPTS_PATH}'")

    hr_lang_ckpts_folder = Path(
        ALL_CKPTS_PATH,
        f"{prefix_training_folder}{hr_lang}"
    )
    hr_lang_checkpoints = list(hr_lang_ckpts_folder.glob("**/*.pt"))
    if len(hr_lang_checkpoints) == 0:
        raise Exception(
            f"There are no checkpoints for '{hr_lang}' in '{hr_lang_ckpts_folder}'"
        )

    start_time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(LOG_SAVE_FOLDER, exist_ok=True)

    all_results = {}

    for checkpoint_path in tqdm(hr_lang_checkpoints):
        clear_memory()

        init = mt5PerplexityExperiments(
            device=device, model_id=model_id, checkpoint_path=checkpoint_path
        )
        for lr_lang in tqdm(LR_LANGS, desc="Langs processing"):
            clear_memory()

            test_dir_path = Path(XL_DATASET_PATH, lr_lang)
            try:
                test_perp = init.testing(test_dir=test_dir_path)
            except:
                test_perp = None

            str_checkpoint_path = str(checkpoint_path)
            if str_checkpoint_path not in all_results:
                all_results[str_checkpoint_path] = {}

            if lr_lang not in all_results[str_checkpoint_path]:
                all_results[str_checkpoint_path][lr_lang] = test_perp

            with open(Path(LOG_SAVE_FOLDER, f"log_{start_time}.json"), "w+") as f:
                json.dump(all_results, f)


if __name__ == "__main__":
    Fire(main)
