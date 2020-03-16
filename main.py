import torch
import numpy as np
from models import CCModel
from data import get_ASR_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='run training')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
args = parser.parse_args()
train = args.train
restart = args.restart
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
if restart: trainer.load_checkpoint()

if not train:
	from data import CollateWavsASR
	import matplotlib.pyplot as plt
	c = CollateWavsASR()
	indices = [0,1]
	b = [ train_dataset.__getitem__(idx) for idx in indices]
	batch = c.__call__(b)
	x,y,T,U,idxs = batch
	model.eval()
	encoded, predicted, diff = model.autoregressive_model(x,T)
	fbank = model.autoregressive_model.compute_fbank((x,T))
	plt.subplot(2,1,1); plt.imshow(predicted[0].cpu().detach().transpose(0,1)); plt.subplot(2,1,2); plt.imshow(fbank[0].cpu().detach().transpose(0,1)); plt.show()
	loss, p_big, I_big = model(x,y,T,U, alpha=0)
	p_big = p_big.cpu()
	plt.plot(p_big[0].detach()); plt.show()

#from data import CollateWavsASR
#import matplotlib.pyplot as plt
#c = CollateWavsASR()
#indices = [0,1]
#b = [ train_dataset.__getitem__(idx) for idx in indices]
#batch = c.__call__(b)
#x,y,T,U,idxs = batch
#encoded, predicted, diff = model.autoregressive_model(x,T)
#fbank = model.autoregressive_model.compute_fbank((x,T))
#plt.subplot(2,1,1); plt.imshow(predicted[0].cpu().detach().transpose(0,1)); plt.subplot(2,1,2); plt.imshow(fbank[0].cpu().detach().transpose(0,1)); plt.show()
#t = 3; ((diff[0].cpu()**2).mean(1).detach() > t).sum(); plt.plot((diff[0].cpu()**2).mean(1).detach() > t); plt.show()

# Train the final model
if train:
	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_WER, train_loss = trainer.train(train_dataset)
		if epoch % 10 == 0:
			valid_WER, valid_loss = trainer.test(valid_dataset, set="valid")
			print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
			print("train WER: %.2f| train loss: %.2f| valid WER: %.2f| valid loss: %.2f\n" % (train_WER, train_loss, valid_WER, valid_loss) )
			trainer.save_checkpoint(WER=valid_WER)

	trainer.load_best_model()
	test_WER, test_loss = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test WER: %.2f| test loss: %.2f \n" % (test_WER, test_loss) )
