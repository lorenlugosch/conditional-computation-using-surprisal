from subprocess import call
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

experiments_folder = "experiments"
#base_config_name = "mini-librispeech"
base_config_name = "timit"
base_config_path = os.path.join(experiments_folder, base_config_name + ".cfg")

class Experiment:
	def __init__(self, use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, randomize, lmbd, big_only=False, small_only=False):
		self.use_AR_features = use_AR_features
		self.surprisal_triggered_sampling_during_training = surprisal_triggered_sampling_during_training
		self.num_seeds = num_seeds
		self.experiments_folder = experiments_folder
		self.base_config_name = base_config_name
		self.base_config_path = base_config_path
		self.big_only = big_only
		self.small_only = small_only
		self.randomize = randomize
		self.lmbd = lmbd
		if self.big_only:
			self.surprisal_triggered_sampling_during_training = False
			self.probability_of_sampling_big_during_training=1.0
			self.probability_of_sampling_big_during_testing=1.0
		if self.small_only:
			self.surprisal_triggered_sampling_during_training = False
			self.probability_of_sampling_big_during_training=0.0
			self.probability_of_sampling_big_during_testing=0.0
		if not self.big_only and not self.small_only:
			self.probability_of_sampling_big_during_training=0.5
			self.probability_of_sampling_big_during_testing=0.5

	def run(self):
		if self.big_only:
			self.name = self.base_config_name + "_big_only"
		if self.small_only:
			self.name = self.base_config_name + "_small_only"
		if not self.big_only and not self.small_only:
			self.name = self.base_config_name + "_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training)) + str(self.randomize) + str(self.lmbd)
		for seed in range(1,self.num_seeds+1):
			# Create config file for this experiment
			experiment_name = self.name + "_seed=%d" % (seed)
			experiment_config_path = os.path.join(experiments_folder, experiment_name + ".cfg")
			with open(self.base_config_path, "r") as f:
				lines = f.readlines()

			seed_line_index = ["seed=" in line for line in lines].index(True)
			lines[seed_line_index] = "seed=" + str(seed) + "\n" 
			folder_line_index = ["folder=" in line for line in lines].index(True)
			lines[folder_line_index]="folder="+os.path.join(self.experiments_folder, experiment_name)+"\n"
			use_AR_features_line = ["use_AR_features=" in line for line in lines].index(True)
			lines[use_AR_features_line]="use_AR_features="+str(self.use_AR_features)+"\n"
			surprisal_triggered_sampling_during_training_line = ["sample_based_on_surprisal_during_training=" in line for line in lines].index(True)
			lines[surprisal_triggered_sampling_during_training_line]="sample_based_on_surprisal_during_training="+str(self.surprisal_triggered_sampling_during_training)+"\n"
			probability_of_sampling_big_during_training_line = ["probability_of_sampling_big_during_training=" in line for line in lines].index(True)
			lines[probability_of_sampling_big_during_training_line] = "probability_of_sampling_big_during_training="+str(self.probability_of_sampling_big_during_training)+"\n"
			probability_of_sampling_big_during_testing_line = ["probability_of_sampling_big_during_testing=" in line for line in lines].index(True)
			lines[probability_of_sampling_big_during_testing_line] = "probability_of_sampling_big_during_testing="+str(self.probability_of_sampling_big_during_testing)+"\n"
			lmbd_line = ["lmbd=" in line for line in lines].index(True)
			lines[lmbd_line] = "lmbd="+ str(self.lmbd)

			with open(experiment_config_path, "w") as f:
				f.writelines(lines)

			# Run the experiment
			cmd = "python main.py --train --config_path=\""+ experiment_config_path +"\""
			if self.randomize: cmd += " --randomize"
			call(cmd, shell=True)

	def get_results(self, column, set):
		#self.name = self.base_config_name + "_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training))
		if self.big_only:
			self.name = self.base_config_name + "_big_only"
		if self.small_only:
			self.name = self.base_config_name + "_small_only"
		if not self.big_only and not self.small_only:
			self.name = self.base_config_name + "_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training)) + str(self.randomize) + str(self.lmbd)

		trials = []
		for seed in range(1,self.num_seeds+1):
			experiment_name = self.name + "_seed=%d" % (seed)
			experiment_results_path = os.path.join(experiments_folder, experiment_name, "training/log.csv")
			df = pd.read_csv(experiment_results_path)
			trials.append(df.loc[df["set"] == set][column].to_numpy())

		trials = np.stack(trials)
		return trials.mean(0), np.sqrt(trials.var(0))

experiments = []
num_seeds = 5

# other
use_AR_features = True
surprisal_triggered_sampling_during_training = True
for randomize in [False, True]:
	for lmbd in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
		experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, randomize, lmbd)
		experiments.append(experiment)

# run experiments
for experiment in experiments:
	experiment.run()

# print test results
for experiment in experiments:
	test_loss_mean_random, test_loss_std_random = experiment.get_results(column="loss", set="test")
	test_WER_mean_random, test_WER_std_random = experiment.get_results(column="WER", set="test")
	test_FLOPs_mean_random, _ = experiment.get_results(column="FLOPs_mean", set="test")
	print(experiment.name)
	print("loss: %.2f $\pm$ %.2f" % (test_loss_mean_random, test_loss_std_random))
	print("WER: %.2f\%% $\pm$ %.2f\%%" % (test_WER_mean_random*100, test_WER_std_random*100))
	print("FLOPs: %d" % (test_FLOPs_mean_random))
