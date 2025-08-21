# COVID-19 NLP Project

This repository contains two Jupyter notebooks implementing **Part A** (Exploratory Data Analysis & text preprocessing) and **Part B** (model fine‑tuning and compression) for the COVID‑19 tweet sentiment classification assignment.

## Structure

- **PartA_COVID19_EDA.ipynb** – Performs a thorough EDA on the Kaggle *COVID‑19 NLP Text Classification* dataset.  It examines sentiment distribution, tweet length and temporal patterns, shows common words using frequency and TF‑IDF analyses, and cleans the raw tweets.  The notebook includes reflections on the choices made and exports a cleaned CSV ready for modeling.
- **Part_B_Covid_tweets.ipynb** – Fine‑tunes two pretrained models (DistilBERT and RoBERTa) on the cleaned tweets using two approaches: a custom PyTorch training loop and the Hugging Face `Trainer` API.  Hyperparameters are optimized using **Optuna** and all experiments are logged to **Weights & Biases** (W&B).  Three model compression techniques—dynamic quantization, unstructured pruning and knowledge distillation—are then applied and the compressed models are evaluated.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), [Datasets](https://huggingface.co/docs/datasets), [Accelerate](https://huggingface.co/docs/accelerate), [Evaluate](https://huggingface.co/docs/evaluate), [Optuna](https://optuna.org), [Weights & Biases](https://wandb.ai), [scikit‑learn](https://scikit-learn.org), and [kagglehub](https://github.com/Kaggle/kagglehub).
- A **Kaggle** API key configured in your environment (for downloading the dataset via `kagglehub.dataset_download`).
- A **W&B** API key if you wish to log experiments (you can disable logging in the notebook by setting `use_wandb=False`).

You can install the required Python packages with:

```bash
pip install transformers datasets accelerate evaluate optuna wandb scikit-learn kagglehub
```

## Running the Notebooks

1. Ensure your Kaggle API credentials are available (the `kaggle.json` file should be in `~/.kaggle`).
2. (Optional) Authenticate with W&B using `wandb login` and your API key.
3. Open **PartA_COVID19_EDA.ipynb** in Jupyter Notebook or JupyterLab and run all cells.  This notebook performs EDA and outputs a cleaned dataset ready for modeling.
4. Open **Part_B_Covid_tweets.ipynb** and run all cells.  This notebook fine‑tunes DistilBERT and RoBERTa using both a manual training loop and the Hugging Face `Trainer` API, tunes hyperparameters with Optuna, logs results to W&B, and applies three compression techniques.

## Notes

- The notebooks are designed to be reproducible and include detailed comments, reasoning, and reflections on both successful and unsuccessful experiments, as required by the assignment.
- During Optuna tuning, checkpoints are saved for the best trial and loaded for evaluation.  Ensure there is enough disk space to store these checkpoints if you run many trials.
- The compression section evaluates the quantized, pruned and distilled models and reports their metrics to help compare trade‑offs between size and accuracy.

