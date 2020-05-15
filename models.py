import torch
import torchaudio
import numpy as np
import sys
sys.path.append("..") # this should be where the autoregressive model repo is located
import os
import matplotlib.pyplot as plt
import ctcdecode
from autoregressive_models.models import AutoregressiveModel, RNNOutputSelect

def count_params(model):
	return sum([p.numel() for p in model.parameters()])

class CCModel(torch.nn.Module):
	def __init__(self, config):
		super(CCModel, self).__init__()
		self.blank_index = config.num_tokens
		self.num_outputs = config.num_tokens + 1

		self.autoregressive_model = AutoregressiveModel(config)
		if config.frame_skipping:
			self.autoregressive_model.load_state_dict(torch.load("../autoregressive_models/experiments/80_mel_frame_skipping/training/model_state.pth"))
		else:
			self.autoregressive_model.load_state_dict(torch.load("../autoregressive_models/experiments/80_mel_no_frame_skipping/training/model_state.pth"))
		print("Number of params in autoregressive model:", count_params(self.autoregressive_model))
		for param in self.autoregressive_model.parameters():
			param.requires_grad = False

		self.controller = Controller()
		self.use_AR_features = config.use_AR_features
		if self.use_AR_features:
			self.encoder_out_dim = self.autoregressive_model.encoder.out_dim
		else:
			self.encoder_out_dim = config.num_mel_bins
		self.prenet_out_dim = 512
		self.main_out_dim = self.prenet_out_dim
		self.postnet_out_dim = self.prenet_out_dim

		self.prenet = torch.nn.Sequential(
				torch.nn.GRU(input_size=self.encoder_out_dim, hidden_size=self.prenet_out_dim // 2, batch_first=True, bidirectional=True),
				RNNOutputSelect(),
				torch.nn.Dropout(0.5),
		)
		print("Number of params in prenet:", count_params(self.prenet))

		self.small_model = torch.nn.Sequential(
				torch.nn.Linear(self.prenet_out_dim, self.main_out_dim),
				torch.nn.LeakyReLU(0.125),
		)
		print("Number of params in small model:", count_params(self.small_model))

		self.big_model = torch.nn.Sequential(
				torch.nn.Linear(self.prenet_out_dim, config.big_model_dim),
				torch.nn.LeakyReLU(0.125),
				torch.nn.Linear(config.big_model_dim, self.main_out_dim),
				torch.nn.LeakyReLU(0.125),
		)
		print("Number of params in big model:", count_params(self.big_model))

		self.postnet = torch.nn.Sequential(
				torch.nn.GRU(input_size=self.main_out_dim, hidden_size=self.postnet_out_dim // 2, batch_first=True, bidirectional=True),
				RNNOutputSelect(),
				torch.nn.Dropout(0.5),
				torch.nn.Linear(self.postnet_out_dim, self.num_outputs),
				torch.nn.LogSoftmax(dim=2)
		)
		print("Number of params in postnet:", count_params(self.postnet))

		# for computing the cost of running big and small models:
		self.FLOPs_big = count_params(self.autoregressive_model) + count_params(self.prenet) + count_params(self.big_model) + count_params(self.postnet) - count_params(self.autoregressive_model.regressor) + count_params(self.controller)
		self.FLOPs_small = count_params(self.autoregressive_model) + count_params(self.prenet) + count_params(self.small_model) + count_params(self.postnet) - count_params(self.autoregressive_model.regressor) + count_params(self.controller)

		# beam search
		labels = ["a" for _ in range(self.num_outputs)] # doesn't matter, just need 1-char labels
		self.decoder = ctcdecode.CTCBeamDecoder(labels, blank_id=self.blank_index, beam_width=config.beam_width)

	def compute(self, x, T):
		# run the neural networks
		h, x_hat, diff = self.autoregressive_model(x, T)
		if not self.use_AR_features: # use raw input features
			h = self.autoregressive_model.compute_fbank((x,T))
			h = self.prenet(h)
			out_small = self.small_model(h)
			out_big = self.big_model(h)
		else: # use AR features
			h = self.prenet(h)
			out_small = self.small_model(h[:,1:,:])
			out_big = self.big_model(h[:,1:,:])

		# sample from big or small models
		I_big = self.controller(h[:,1:,:])
		main_out = (1 - I_big) * out_small + I_big * out_big
		out = self.postnet(main_out)
		return out, I_big

	def forward(self, x, y, T, U):
		"""
		returns log probs, p_big, I_big for each example
		"""

		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()
			y = y.cuda()

		out, I_big = self.compute(x, T)

		# compute the log probs
		downsampling_factor = max(T) / out.shape[1]
		T = [round(t / downsampling_factor) for t in T]
		out = out.transpose(0,1) # (N, T, #labels) --> (T, N, #labels)
		log_probs = -torch.nn.functional.ctc_loss(	log_probs=out,
								targets=y,
								input_lengths=T,
								target_lengths=U,
								reduction="none",
								blank=self.blank_index,
								)

		return log_probs, I_big

	def infer(self, x, T=None):
		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		out, I_big = self.compute(x, T)

		# run a beam search
		out = torch.nn.functional.softmax(out, dim=2)
		beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(out)
		decoded = [beam_result[i][0][:out_seq_len[i][0]].tolist() for i in range(len(out))]
		return decoded, I_big

class Controller(torch.nn.Module):
	def __init__(self):
		super(Controller, self).__init__()
		self.linear1 = torch.nn.Linear(512, 80)
		self.relu = torch.nn.LeakyReLU(0.125)
		self.linear2 = torch.nn.Linear(80, 1)
		self.threshold = Threshold()

	def forward(self, h):
		out = self.linear1(h)
		out = self.relu(out)
		out = self.linear2(out)
		I_big = self.threshold.apply(out)
		return I_big

# Identity straight-through estimator, modified from https://discuss.pytorch.org/t/custom-binarization-layer-with-straight-through-estimator-gives-error/4539/3
# https://arxiv.org/pdf/1308.3432.pdf
# https://discuss.pytorch.org/t/how-to-overwrite-a-backwards-pass/65845
class Threshold(torch.autograd.Function):
	@staticmethod
	def forward(self, input):
		return (input > 0).float()

	@staticmethod
	def backward(self, grad_output):
		return grad_output

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
