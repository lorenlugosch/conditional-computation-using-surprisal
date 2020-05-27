import numpy as np
import torch
from tqdm import tqdm # for displaying progress bar
import os
import pandas as pd
from jiwer import wer as compute_WER

class Trainer:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		self.lr = config.lr
		self.lr_period = config.lr_period
		self.gamma = config.gamma
		self.lmbd = config.lmbd
		self.checkpoint_path = os.path.join(self.config.folder, "training")
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_period, gamma=self.gamma)
		self.epoch = 0
		self.df = None
		if torch.cuda.is_available(): self.model.cuda()
		self.best_WER = np.inf

	def load_best_model(self):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "best_model.pth"), map_location=device))

	def save_checkpoint(self, WER):
		try:
			torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "model_state.pth"))
			if WER < self.best_WER:
				self.best_WER = WER
				torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "best_model.pth"))
		except:
			print("Could not save model")

	def log(self, results):
		if self.df is None:
			self.df = pd.DataFrame(columns=[field for field in results])
		self.df.loc[len(self.df)] = results
		self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

	def remove_repeated_silence(self, x):
		x_out = []
		prev = ""
		for xx in x.split():
			if xx != "sil":
				x_out.append(xx)
			else:
				if prev != "sil":
					x_out.append(xx)
			prev = xx
		return " ".join(x_out)

	def train(self, dataset, print_interval=100, log_results=True):
		train_WER = 0
		train_loss = 0
		num_examples = 0
		self.model.train()
		for g in self.optimizer.param_groups:
			print("Current learning rate:", g['lr'])
		FLOPs = []
		for idx, batch in enumerate(tqdm(dataset.loader)):
			x,y,T,U,idxs = batch
			batch_size = len(x)
			log_probs,I_big = self.model(x,y,T,U)
			batch_FLOPs = (1-I_big.mean())*self.model.FLOPs_small + I_big.mean()*self.model.FLOPs_big
			FLOPs += [batch_FLOPs.item()] * batch_size
			loss = -log_probs.mean() + self.lmbd * ((I_big - 0.5)**2).sum()
			if torch.isnan(loss):
				print("nan detected!")
				sys.exit()
			self.optimizer.zero_grad()
			loss.backward()
			clip_value = 5; torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
			self.optimizer.step()
			train_loss += loss.item() * batch_size
			num_examples += batch_size
			if idx % print_interval == 0:
				print("loss: " + str(loss.cpu().data.numpy().item()))
				guesses,_ = self.model.infer(x, T)
				guess = guesses[0][:U[0]]
				guess_decoded = self.remove_repeated_silence(dataset.tokenizer.DecodeIds(guess))
				print("guess:", guess_decoded)
				truth = y[0].cpu().data.numpy().tolist()[:U[0]]
				truth_decoded = self.remove_repeated_silence(dataset.tokenizer.DecodeIds(truth))
				print("truth:", truth_decoded)
				print("WER: ", compute_WER(truth_decoded, guess_decoded))
				print("avg I_big: ", I_big.mean().item())
				print("")

		train_loss /= num_examples
		train_WER /= num_examples
		FLOPs = np.array(FLOPs)
		FLOPs_mean = FLOPs.mean()
		FLOPs_std = FLOPs.std()
		results = {"loss" : train_loss, "WER" : train_WER, "FLOPs_mean" : FLOPs_mean, "FLOPs_std": FLOPs_std, "set": "train"}
		if log_results: self.log(results)
		self.epoch += 1
		return train_WER, train_loss, FLOPs_mean, FLOPs_std

	def test(self, dataset, set, log_results=True, print_results=True):
		test_WER = 0
		test_loss = 0
		num_examples = 0
		self.model.eval()
		FLOPs = []
		for idx, batch in enumerate(dataset.loader):
			x,y,T,U,_ = batch
			batch_size = len(x)
			num_examples += batch_size
			log_probs,I_big = self.model(x,y,T,U)
			loss = -log_probs.mean() + self.lmbd * ((I_big - 0.5)**2).sum()
			test_loss += loss.item() * batch_size
			WERs = []
			guesses,I_big = self.model.infer(x, T)
			batch_FLOPs = (1-I_big.mean())*self.model.FLOPs_small + I_big.mean()*self.model.FLOPs_big
			FLOPs += [batch_FLOPs.item()] * batch_size
			for i in range(batch_size):
				guess = guesses[i][:U[i]]
				truth = y[i].cpu().data.numpy().tolist()[:U[i]]
				guess_decoded = self.remove_repeated_silence(dataset.tokenizer.DecodeIds(guess))
				truth_decoded = self.remove_repeated_silence(dataset.tokenizer.DecodeIds(truth))
				WERs.append(compute_WER(truth_decoded, guess_decoded))
			WER = np.array(WERs).mean()
			test_WER += WER * batch_size
			if print_results:
				print("guess:", guess_decoded)
				print("truth:", truth_decoded)
				print("WER: ", compute_WER(truth_decoded, guess_decoded))
				print("I_big: ", I_big.mean().item())
				print("")

		test_loss /= num_examples
		self.scheduler.step()
		test_WER /= num_examples
		FLOPs = np.array(FLOPs)
		FLOPs_mean = FLOPs.mean()
		FLOPs_std = FLOPs.std()
		results = {"loss" : test_loss, "WER" : test_WER, "FLOPs_mean" : FLOPs_mean, "FLOPs_std": FLOPs_std, "set": set}
		if log_results: self.log(results)
		return test_WER, test_loss, FLOPs_mean, FLOPs_std
