import torch
import numpy as np
from models import CCModel
from data import get_ASR_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='run training')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
args = parser.parse_args()
train = args.train
config_path = args.config_path

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Initialize model
model = CCModel(config=config)
print(model)

# Generate datasets
train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

trainer = Trainer(model=model, config=config)

# just debuggin'
if not train:
	from data import CollateWavsASR
	import matplotlib.pyplot as plt
	c = CollateWavsASR()
	indices = [100,1]
	b = [ train_dataset.__getitem__(idx) for idx in indices]
	batch = c.__call__(b)
	x,y,T,U,idxs = batch
	model.eval()
	encoded, predicted, diff = model.autoregressive_model(x,T)
	fbank = model.autoregressive_model.compute_fbank((x,T))
	loss, p_big, I_big = model(x,y,T,U)
	posteriors = model.get_posteriors(x,T)
	p_big = p_big.cpu()
	print(p_big.shape)
	"""
	plt.subplot(4,1,1); plt.imshow(predicted[0].cpu().detach().transpose(0,1), aspect="auto"); 
	plt.subplot(4,1,2); plt.imshow(fbank[0].cpu().detach().transpose(0,1), aspect="auto"); 
	plt.subplot(4,1,3); plt.plot(p_big[0].detach()); plt.xlim(0,predicted.shape[1]); 
	plt.subplot(4,1,4); plt.imshow(posteriors[0].cpu().detach().transpose(0,1), aspect="auto"); 
	plt.show()
	"""

if train:
	print("Training the controller...")
	trainer.train_controller(train_dataset)
	print("Done.")
	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_WER, train_loss, train_FLOPs_mean, train_FLOPs_std = trainer.train(train_dataset)
		if epoch % config.validation_period == 0:
			model.sample_based_on_surprisal_during_testing = False
			valid_WER_random, valid_loss_random, valid_FLOPs_mean_random, valid_FLOPs_std_random = trainer.test(valid_dataset, set="valid")
			trainer.save_checkpoint(WER=valid_WER_random, sampling_method="random")

			model.sample_based_on_surprisal_during_testing = True
			valid_WER_surprisal, valid_loss_surprisal, valid_FLOPs_mean_surprisal, valid_FLOPs_std_surprisal = trainer.test(valid_dataset, set="valid")
			trainer.save_checkpoint(WER=valid_WER_surprisal, sampling_method="surprisal")
		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("train WER: %.2f| train loss: %.2f| train FLOPs: %d" % (train_WER * 100, train_loss, train_FLOPs_mean))
		print("valid WER: %.2f| valid loss: %.2f| valid FLOPs: %d (random sampling)" % (valid_WER_random * 100, valid_loss_random, valid_FLOPs_mean_random) )
		print("valid WER: %.2f| valid loss: %.2f| valid FLOPs: %d (surprisal sampling)\n" % (valid_WER_surprisal * 100, valid_loss_surprisal, valid_FLOPs_mean_surprisal) )

	trainer.load_best_model(sampling_method="random")
	model.sample_based_on_surprisal_during_testing = False
	test_WER_random, test_loss_random, test_FLOPs_mean_random, test_FLOPs_std_random = trainer.test(test_dataset, set="test")

	trainer.load_best_model(sampling_method="surprisal")
	model.sample_based_on_surprisal_during_testing = True
	test_WER_surprisal, test_loss_surprisal, test_FLOPs_mean_surprisal, test_FLOPs_std_surprisal = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test WER: %.2f| test loss: %.2f| test FLOPs: %d (random sampling)" % (test_WER_random * 100, test_loss_random, test_FLOPs_mean_random) )
	print("test WER: %.2f| test loss: %.2f| test FLOPs: %d (surprisal sampling)\n" % (test_WER_surprisal * 100, test_loss_surprisal, test_FLOPs_mean_surprisal) )
