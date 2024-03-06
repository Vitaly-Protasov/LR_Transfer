# Cross-lingual transfer between languages

## Getting started
1. Install jax libs:
    ```bash
    pip install jax==0.3.22 jaxlib==0.3.22 -f https://storage.googleapis.com/jax-releases/jax_releases.html
    ```

2. Install dependecies:
    ```bash
    pip install -r requirements.txt
    ```

## Pipeline of experiments

### Dataset
1. Dataset is available here: [link]()
2. See [resource_processing/](resource_processing/) for information regarding resources and scripts that were used during dataset collection:
    * [hzsk_processing](resource_processing/hzsk_processing.py)
    * [mc4_processing](resource_processing/mc4_processing.py)
    * [ud_processing](resource_processing/ud_processing.py)
    * [wiki_processing](wiki_processing/ud_processing.py)
    * [vk_hse_processing](wiki_processing/ud_processing.py)
    
### MLM experiments
1. Continued pretraining on high-resource languages: [hr_training](mt5_experiments/hr_training.py)
2. Evaluation of checkpoints on low-resource languages: [lr_evaluation](mt5_experiments/lr_evaluation.py)
3. Vizualization of obtained results: [notebook1](notebooks/mlm-statistic.ipynb), [notebook2](notebooks/TL%20visualization.ipynb)

### Analysis using language and data features
1. Calculation of token intersection between languages: [notebook](notebooks/Tokens%20intersection.ipynb)
2. Usage of WALS features: [notebook](notebooks/wals_feat.ipynb)

### Downstream evaluation
1. Machine translation: #TODO
2. POS tagging: #TODO