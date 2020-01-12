import torch
import torchaudio
import numpy as np
import sys
sys.path.append("..") # this should be where the autoregressive model repo is located
import os
import matplotlib.pyplot as plt
import ctcdecode
from autoregressive_models.models import AutoregressiveModel, RNNOutputSelect

class CTCModel(torch.nn.Module):
	def __init__(self, config):
		super(CTCModel, self).__init__()
		self.blank_index = config.num_tokens
		self.num_outputs = config.num_tokens + 1

		self.autoregressive_model = AutoregressiveModel(config)
		self.autoregressive_model.load_state_dict(torch.load("../autoregressive_models/experiments/80_mel/training/model_state.pth"))
		AR_out_dim = self.autoregressive_model.encoder.out_dim
		for param in self.autoregressive_model.parameters():
			param.requires_grad = False

		self.controller = Controller()
		self.subsequent_dim = 256
		self.prenet = torch.nn.Sequential(
				#Conv(in_dim=AR_out_dim, out_dim=self.subsequent_dim, filter_length=3, stride=1),
				torch.nn.GRU(input_size=AR_out_dim, hidden_size=int(self.subsequent_dim/2), batch_first=True, bidirectional=True),
				RNNOutputSelect(),
				torch.nn.Dropout(0.25),
				torch.nn.Linear(self.subsequent_dim, self.subsequent_dim),
				torch.nn.LeakyReLU(0.125),
		)
		self.small_model = torch.nn.Sequential(
				torch.nn.Linear(self.subsequent_dim, self.num_outputs),
				torch.nn.LogSoftmax(dim=2)
		)
		self.big_model = torch.nn.Sequential(
				torch.nn.Linear(self.subsequent_dim, config.big_model_dim),
				torch.nn.LeakyReLU(0.125),
				torch.nn.Dropout(0.25),
				torch.nn.Linear(config.big_model_dim, self.num_outputs),
				torch.nn.LogSoftmax(dim=2)
		)

		# beam search
		labels = ["a" for _ in range(self.num_outputs)] # doesn't matter, just need 1-char labels
		self.decoder = ctcdecode.CTCBeamDecoder(labels, blank_id=self.blank_index, beam_width=config.beam_width)

	def forward(self, x, y, T, U, alpha=0.75):
		"""
		returns log probs for each example
		"""

		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()
			y = y.cuda()

		# run the neural networks
		h, x_hat, diff = self.autoregressive_model(x, T)
		h = self.prenet(h)

		p_big = self.controller(diff, alpha=alpha)
		out_small = self.small_model(h[:,1:,:])
		out_big = self.big_model(h[:,1:,:])
		I_big = torch.distributions.binomial.Binomial(1, p_big).sample() #(p_big > 0.5).float() # select one model

		encoder_out = (1 - I_big) * out_small + I_big * out_big

		# run the forward algorithm to compute the log probs
		downsampling_factor = max(T) / encoder_out.shape[1]
		T = [round(t / downsampling_factor) for t in T]
		encoder_out = encoder_out.transpose(0,1) # (N, T, #labels) --> (T, N, #labels)
		log_probs = -torch.nn.functional.ctc_loss(	log_probs=encoder_out,
								targets=y,
								input_lengths=T,
								target_lengths=U,
								reduction="none",
								blank=self.blank_index)
		return log_probs, p_big, I_big

	def infer(self, x, T=None, alpha=0.75):
		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		# run the neural network
		h, x_hat, diff = self.autoregressive_model(x, T)
		h = self.prenet(h)

		p_big = self.controller(diff, alpha=alpha)
		out_small = self.small_model(h[:,1:,:])
		out_big = self.big_model(h[:,1:,:])
		I_big = torch.distributions.binomial.Binomial(1, p_big).sample()

		out = (1 - I_big) * out_small + I_big * out_big

		# run a beam search
		out = torch.nn.functional.softmax(out, dim=2)
		beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(out)
		decoded = [beam_result[i][0][:out_seq_len[i][0]].tolist() for i in range(len(out))]
		return decoded


class Controller(torch.nn.Module):
	def __init__(self):
		super(Controller, self).__init__()
		self.bias = torch.nn.Parameter(torch.tensor([-100.0]), requires_grad=False)

	def forward(self, diff, alpha):
		error = (diff ** 2).sum(2).unsqueeze(2)
		p_big = torch.sigmoid(error + self.bias)

		# alpha = 1 --> p_big is uniformly distributed
		# alpha = 0 --> p_big is learned, deterministic
		noise = torch.rand(p_big.shape).to(p_big.device)
		return alpha* noise + (1-alpha) * p_big

class Conv(torch.nn.Module):
	def __init__(self, in_dim, out_dim, filter_length, stride):
		super(Conv, self).__init__()
		self.conv = torch.nn.Conv1d(	in_channels=in_dim,
						out_channels=out_dim,
						kernel_size=filter_length,
						stride=stride
		)
		self.filter_length = filter_length

	def forward(self, x):
		out = x.transpose(1,2)
		left_padding = int(self.filter_length/2)
		right_padding = int(self.filter_length/2)
		out = torch.nn.functional.pad(out, (left_padding, right_padding))
		out = self.conv(out)
		out = out.transpose(1,2)
		return out
