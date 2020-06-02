# conditional-computation-using-surprisal

This repo contains the experiment code for our paper, "Surprisal-Triggered Conditional Computation with Neural Networks".

## Requirements

- PyTorch
- torchaudio
- ctcdecode
- tqdm
- The `autoregressive-models` repo (get [here](https://github.com/lorenlugosch/autoregressive-models), clone into a folder adjacent to this one, and rename to `autoregressive_models` (needed for import to work))

We used a single Tesla K80 GPU for our experiments. Training a single model takes about 1 hour on this GPU. Each experiment is run by training models with 5 random seeds, so a single experiment will require about 5 hours.

## Datasets

The code has been tested for the following datasets:

- Mini-LibriSpeech
- TIMIT

Mini-LibriSpeech can be downloaded for free [here](https://www.openslr.org/31/). TIMIT must be purchased from LDC.

After you've downloaded the dataset, change the `base_path` field in `mini-librispeech.cfg` and `timit.cfg` to point to where your folder the dataset folder is located, and move the `*.csv` files from `experiments/mini-librispeech-csv` or `experiments/timit-csv` into the dataset folder. (Note that `valid_data.csv` and `test_data.csv` are the same for Mini-LibriSpeech, as it has only train and test sets and no validation set.)

## Running the experiments

Execute `python run_experiments.py`. (By default, this will run the ablation experiment on Mini-LibriSpeech. Change the base path in `run_experiments.py` if you want to use TIMIT.)

To run the baseline with a learned controller, switch to the `learnedcontroller` branch and execute `python run_experiments.py`.

## Training a single model

Execute `python main.py --train --config_path="<path to your .cfg>"` (e.g., `experiments/mini-librispeech.cfg`)
