from subprocess import call
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

experiments_folder = "experiments"
base_config_name = "timit" #"mini-librispeech"
base_config_path = os.path.join(experiments_folder, base_config_name + ".cfg")

class Experiment:
	def __init__(self, use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, big_only=False, small_only=False):
		self.use_AR_features = use_AR_features
		self.surprisal_triggered_sampling_during_training = surprisal_triggered_sampling_during_training
		self.num_seeds = num_seeds
		self.experiments_folder = experiments_folder
		self.base_config_name = base_config_name
		self.base_config_path = base_config_path
		self.big_only = big_only
		self.small_only = small_only
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
			self.name = self.base_config_name + "_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training))
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

			with open(experiment_config_path, "w") as f:
				f.writelines(lines)

			# Run the experiment
			call("python main.py --train --config_path=\""+ experiment_config_path +"\"", shell=True)

	def get_results(self, column, set):
		#self.name = self.base_config_name + "_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training))
		if self.big_only:
			self.name = self.base_config_name + "_big_only"
		if self.small_only:
			self.name = self.base_config_name + "_small_only"
		if not self.big_only and not self.small_only:
			self.name = self.base_config_name + "_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training))

		trials = []
		for seed in range(1,self.num_seeds+1):
			experiment_name = self.name + "_seed=%d" % (seed)
			experiment_results_path = os.path.join(experiments_folder, experiment_name, "training/log.csv")
			df = pd.read_csv(experiment_results_path)
			trials.append(df[column].loc[df["set"] == set].to_numpy())

		trials = np.stack(trials)
		return trials.mean(0), np.sqrt(trials.var(0))

experiments = []
num_seeds = 5

"""
use_AR_features = True; surprisal_triggered_sampling_during_training=False
# big only
experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, big_only=True)
experiments.append(experiment)
# small only
experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, small_only=True)
experiments.append(experiment)
"""
# other
for use_AR_features in [False, True]:
	for surprisal_triggered_sampling_during_training in [False, True]:
		experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path)
		experiments.append(experiment)

# run experiments
for experiment in experiments:
	experiment.run()

# plot results
"""
for experiment in experiments:
	valid_loss_mean, valid_loss_std = experiment.get_results(column="WER", set="valid")
	num_epochs = len(valid_loss_mean)
	color = "green" if experiment.surprisal_triggered_sampling_during_testing else "red"
	linestyle = "-" if experiment.surprisal_triggered_sampling_during_training else "--"
	color = "blue" if experiment.small_only else color
	color = "black" if experiment.big_only else color
	
	if experiment.big_only or experiment.small_only: continue
	
	plt.plot(np.arange(0,num_epochs), valid_loss_mean, linestyle=linestyle, color=color, label=experiment.name)
	upper = valid_loss_mean + valid_loss_std
	lower = valid_loss_mean - valid_loss_std
	plt.fill_between(np.arange(0,num_epochs), lower, upper, facecolor=color, alpha=0.5)
plt.legend()
plt.show()
"""

# print test results
for experiment in experiments:
	test_loss_mean, test_loss_std = experiment.get_results(column="loss", set="test")
	test_WER_mean, test_WER_std = experiment.get_results(column="WER", set="test")
	test_FLOPs_mean, _ = experiment.get_results(column="FLOPs_mean", set="test")
	print(experiment.name)
	print("loss: %.2f $\pm$ %.2f" % (test_loss_mean, test_loss_std))
	print("WER: %.2f\%% $\pm$ %.2f\%%" % (test_WER_mean*100, test_WER_std*100))
	print("FLOPs: %d" % (test_FLOPs_mean))
