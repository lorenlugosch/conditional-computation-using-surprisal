# conditional-computation-using-surprisal

This repo contains the experiment code for our paper, "Surprisal-Triggered Conditional Computation with Neural Networks".

## Requirements

- PyTorch
- torchaudio
- ctcdecode
- tqdm
- The `autoregressive-models` repo (get [here](https://github.com/lorenlugosch/autoregressive-models), clone into a folder adjacent to this one, and rename to `autoregressive_models` (needed for import to work))

## Datasets

The code has been tested for the following datasets:

- Mini-LibriSpeech
- TIMIT

Mini-LibriSpeech can be downloaded for free [here](https://www.openslr.org/31/). TIMIT must be purchased from LDC.

## Running the experiments

Execute `python run_experiments.py`. (By default, this will run the ablation experiment on Mini-LibriSpeech. Change the base path in `run_experiments.py` if you want to use TIMIT.)

To run the baseline with a learned controller, switch to the `learnedcontroller` branch and execute `python run_experiments.py`.

## Training a single model

Execute `python main.py --train --config_path="<path to your .cfg>"` (e.g., `experiments/mini-librispeech.cfg`)
