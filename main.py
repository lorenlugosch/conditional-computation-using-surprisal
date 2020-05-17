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
parser.add_argument('--randomize', action='store_true', help='initialize using random controller')
args = parser.parse_args()
train = args.train
config_path = args.config_path
randomize = args.randomize

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Initialize model
model = CCModel(config=config)
print(model)
if randomize:
	model.randomize = True
else:
	model.randomize = False

# Generate datasets
train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)
trainer = Trainer(model=model, config=config)

if train:
	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		if epoch > config.num_epochs / 2 and model.randomize:
			print("switching from random to learned controller")
			model.randomize = False
		train_WER, train_loss, train_FLOPs_mean, train_FLOPs_std = trainer.train(train_dataset)
		if epoch % config.validation_period == 0:
			valid_WER, valid_loss, valid_FLOPs_mean, valid_FLOPs_std = trainer.test(valid_dataset, set="valid")
			trainer.save_checkpoint(WER=valid_WER)
		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("train WER: %.2f| train loss: %.2f| train FLOPs: %d" % (train_WER * 100, train_loss, train_FLOPs_mean))
		print("valid WER: %.2f| valid loss: %.2f| valid FLOPs: %d" % (valid_WER * 100, valid_loss, valid_FLOPs_mean) )

	trainer.load_best_model()
	test_WER, test_loss, test_FLOPs_mean, test_FLOPs_std = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test WER: %.2f| test loss: %.2f| test FLOPs: %d" % (test_WER * 100, test_loss, test_FLOPs_mean) )
