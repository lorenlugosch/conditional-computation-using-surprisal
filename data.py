import torch
import torch.utils.data
import torchaudio
import os
import soundfile as sf
import numpy as np
import configparser
import multiprocessing
import json
import pandas as pd
from subprocess import call
import sentencepiece as spm

class Config:
	def __init__(self):
		pass

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[experiment]
	config.seed=int(parser.get("experiment", "seed"))
	config.folder=parser.get("experiment", "folder")

	# Make a folder containing experiment information
	if not os.path.isdir(config.folder):
		os.mkdir(config.folder)
		os.mkdir(os.path.join(config.folder, "training"))
	call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

	#[model]
	config.num_tokens=int(parser.get("model", "num_tokens"))
	config.num_layers=int(parser.get("model", "num_layers"))
	config.num_hidden=int(parser.get("model", "num_hidden"))
	config.num_mel_bins=int(parser.get("model", "num_mel_bins"))
	config.big_model_dim=int(parser.get("model", "big_model_dim"))
	config.use_AR_features=parser.get("model", "use_AR_features")=="True"
	config.frame_skipping=parser.get("model", "frame_skipping")=="True"
	config.controller_mean=float(parser.get("model", "controller_mean"))
	config.controller_var=float(parser.get("model", "controller_var"))
	#config.tokenizer_training_text_path=parser.get("model", "tokenizer_training_text_path")

	#[training]
	config.validation_period=int(parser.get("training", "validation_period"))
	config.base_path=parser.get("training", "base_path")
	config.lr=float(parser.get("training", "lr"))
	config.lr_period=int(parser.get("training", "lr_period"))
	config.gamma=float(parser.get("training", "gamma"))
	config.batch_size=int(parser.get("training", "batch_size"))
	config.num_epochs=int(parser.get("training", "num_epochs"))
	config.deterministic_train=parser.get("training", "deterministic_train")=="True"
	config.deterministic_test=parser.get("training", "deterministic_test")=="True"
	config.sample_based_on_surprisal_during_training=parser.get("training", "sample_based_on_surprisal_during_training")=="True"
	# if sampling not based on surprisal:
	config.probability_of_sampling_big_during_training=float(parser.get("training", "probability_of_sampling_big_during_training"))
	config.probability_of_sampling_big_during_testing=float(parser.get("training", "probability_of_sampling_big_during_testing"))

	#[inference]
	config.beam_width=int(parser.get("inference", "beam_width"))

	return config

def get_ASR_datasets(config):
	"""
	config: Config object (contains info about model and training)
	"""
	base_path = config.base_path

	# Get dfs
	train_df = pd.read_csv(os.path.join(base_path, "train_data.csv")) #"train-clean-100.csv"))
	valid_df = pd.read_csv(os.path.join(base_path, "valid_data.csv"))
	test_df = pd.read_csv(os.path.join(base_path, "test_data.csv"))

	phoneme_df = pd.read_csv(os.path.join(base_path, "index-to-phoneme-39.csv"))
	phoneme_to_phoneme_index = {phoneme_df.phoneme[i]:int(phoneme_df.index[i]) for i in range(len(phoneme_df)) }
	config.phoneme_to_phoneme_index = phoneme_to_phoneme_index

	# Create dataset objects
	train_dataset = ASRDataset(train_df, config)
	valid_dataset = ASRDataset(valid_df, config)
	test_dataset = ASRDataset(test_df, config)

	return train_dataset, valid_dataset, test_dataset

class PhonemeTokenizer:
	def __init__(self, phoneme_to_phoneme_index):
		self.phoneme_to_phoneme_index = phoneme_to_phoneme_index
		self.phoneme_index_to_phoneme = {v: k for k, v in self.phoneme_to_phoneme_index.items()}

	def EncodeAsIds(self, phoneme_string):
		return [self.phoneme_to_phoneme_index[p] for p in phoneme_string.split()]

	def DecodeIds(self, phoneme_ids):
		return " ".join([self.phoneme_index_to_phoneme[id] for id in phoneme_ids])

class ASRDataset(torch.utils.data.Dataset):
	def __init__(self, df, config):
		"""
		df: dataframe of wav file paths and transcripts
		config: Config object (contains info about model and training)
		"""
		# dataframe with wav file paths, transcripts
		self.df = df
		self.base_path = config.base_path
		self.tokenizer = PhonemeTokenizer(config.phoneme_to_phoneme_index)

		self.loader = torch.utils.data.DataLoader(self, batch_size=config.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateWavsASR())

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		x, fs = sf.read(os.path.join(self.base_path, self.df.path[idx]))
		"""
		if not self.tokenizer_sampling: y = self.tokenizer.EncodeAsIds(self.df.transcript[idx])
		if self.tokenizer_sampling: y = self.tokenizer.SampleEncodeAsIds(self.df.transcript[idx], -1, 0.1)
		"""
		y = self.tokenizer.EncodeAsIds(self.df["phonemes_39"][idx])
		return (x, y, idx)

class CollateWavsASR:
	def __init__(self):
		self.max_length = 500000

	def __call__(self, batch):
		"""
		batch: list of tuples (input wav, output labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; y = []; idxs = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_,idx = batch[index]

			# throw away large audios
			if len(x_) < self.max_length:
				x.append(torch.tensor(x_).float())
				y.append(torch.tensor(y_).long())
				idxs.append(idx)

		batch_size = len(idxs) # in case we threw some away

		# pad all sequences to have same length
		T = [len(x_) for x_ in x]
		T_max = max(T)
		U = [len(y_) for y_ in y]
		U_max = max(U)
		for index in range(batch_size):
			x_pad_length = (T_max - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length))

			y_pad_length = (U_max - len(y[index]))
			y[index] = torch.nn.functional.pad(y[index], (0,y_pad_length), value=-1)

		x = torch.stack(x)
		y = torch.stack(y)

		return (x,y,T,U,idxs)
