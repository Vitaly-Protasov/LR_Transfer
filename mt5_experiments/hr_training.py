from fire import Fire
from typing import List, Union

from tl_experiments import mt5PerplexityExperiments
from tl_utils import clear_memory


def main(
    dataset_paths: Union[str, List[str]],
    device: str = "cuda:0",
    model_id: str = "google/mt5-base",
    per_device_batch_size: int = 8,
    n_epochs: int = 2,
    max_dataset_len: int = 500000,
    save_checkpoints: bool = True,
    num_runs_per_dataset: int = 5,
):
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    for dataset_folder in dataset_paths:
        for _ in range(num_runs_per_dataset):
            clear_memory()
            init = mt5PerplexityExperiments(device=device, model_id=model_id)
            init.training(
                train_valid_dir=dataset_folder,
                per_device_batch_size=per_device_batch_size,
                n_epochs=n_epochs,
                max_dataset_len = max_dataset_len,
                save_checkpoints = save_checkpoints
            )


if __name__ == "__main__":
    Fire(main)
