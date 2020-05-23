from subprocess import call
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

experiments_folder = "experiments"
#base_config_name = "mini-librispeech"
#base_config_name = "timit"
base_config_name = "timit_0.65"
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

	def get_results(self, column, set, surprisal_triggered):
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
			trials.append(df.loc[df["set"] == set].loc[df["surprisal-triggered"] == surprisal_triggered][column].to_numpy())

		trials = np.stack(trials)
		return trials.mean(0), np.sqrt(trials.var(0))

experiments = []
num_seeds = 5
# other
for use_AR_features in [True]:
	for surprisal_triggered_sampling_during_training in [False, True]:
		experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path)
		experiments.append(experiment)
use_AR_features = True; surprisal_triggered_sampling_during_training=False
# big only
experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, big_only=True)
experiments.append(experiment)
# small only
experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, num_seeds, experiments_folder, base_config_name, base_config_path, small_only=True)
experiments.append(experiment)

# run experiments
for experiment in experiments:
	experiment.run()
"""
# plot results
for experiment in experiments:
	for surprisal_triggered_during_testing in [False, True]:
		if experiment.big_only or experiment.small_only: continue
		valid_loss_mean, valid_loss_std = experiment.get_results(column="WER", set="valid", surprisal_triggered=surprisal_triggered_during_testing)
		num_epochs = len(valid_loss_mean)
		color = "blue" if surprisal_triggered_during_testing and experiment.surprisal_triggered_sampling_during_training else "red"
		linestyle = "-" #if experiment.surprisal_triggered_sampling_during_training else "--"
		s = "surprisal (train)="
		s += str(experiment.surprisal_triggered_sampling_during_training)
		s += ", surprisal (test)="
		s += str(surprisal_triggered_during_testing)
		plt.plot(np.arange(0,num_epochs), valid_loss_mean, linestyle=linestyle, color=color, label=s)
		upper = valid_loss_mean + valid_loss_std
		lower = valid_loss_mean - valid_loss_std
		plt.fill_between(np.arange(0,num_epochs), lower, upper, facecolor=color, alpha=0.5)
plt.legend()
plt.title("Validation PER (TIMIT)")
plt.show()
"""

# create .dat
for experiment in experiments:
	for surprisal_triggered_during_testing in [False, True]:
		if experiment.big_only or experiment.small_only: continue
		valid_loss_mean, valid_loss_std = experiment.get_results(column="WER", set="valid", surprisal_triggered=surprisal_triggered_during_testing)
		with open(experiment.name + str(surprisal_triggered_during_testing) + ".dat", "w") as f:
			f.write("x y err\n")
			interval = 5 if base_config_name == "mini-librispeech" else 1
			epochs = np.arange(1, 51, interval)
			for i, epoch in enumerate(epochs):
				f.write(str(epoch) + " " + str(valid_loss_mean[i] * 100) + " " + str(valid_loss_std[i] * 100) + "\n")

# print test results
for experiment in experiments:
	if not experiment.big_only and not experiment.small_only:
		test_loss_mean_random, test_loss_std_random = experiment.get_results(column="loss", set="test", surprisal_triggered=False)
		test_WER_mean_random, test_WER_std_random = experiment.get_results(column="WER", set="test", surprisal_triggered=False)
		test_FLOPs_mean_random, _ = experiment.get_results(column="FLOPs_mean", set="test", surprisal_triggered=False)
		print(experiment.name + "_random")
		print("loss: %.2f $\pm$ %.2f" % (test_loss_mean_random, test_loss_std_random))
		print("WER: %.2f\%% $\pm$ %.2f\%%" % (test_WER_mean_random*100, test_WER_std_random*100))
		print("FLOPs: %d" % (test_FLOPs_mean_random))

		test_loss_mean_surprisal, test_loss_std_surprisal = experiment.get_results(column="loss", set="test", surprisal_triggered=True)
		test_WER_mean_surprisal, test_WER_std_surprisal = experiment.get_results(column="WER", set="test", surprisal_triggered=True)
		test_FLOPs_mean_surprisal, _ = experiment.get_results(column="FLOPs_mean", set="test", surprisal_triggered=True)
		print(experiment.name + "_surprisal")
		print("loss: %.2f $\pm$ %.2f" % (test_loss_mean_surprisal, test_loss_std_surprisal))
		print("WER: %.2f\%% $\pm$ %.2f\%%" % (test_WER_mean_surprisal*100, test_WER_std_surprisal*100))
		print("FLOPs: %d" % (test_FLOPs_mean_surprisal))

	else:
		test_loss_mean_random, test_loss_std_random = experiment.get_results(column="loss", set="test", surprisal_triggered=False)
		test_WER_mean_random, test_WER_std_random = experiment.get_results(column="WER", set="test", surprisal_triggered=False)
		test_FLOPs_mean_random, _ = experiment.get_results(column="FLOPs_mean", set="test", surprisal_triggered=False)
		print(experiment.name)
		print("loss: %.2f $\pm$ %.2f" % (test_loss_mean_random, test_loss_std_random))
		print("WER: %.2f\%% $\pm$ %.2f\%%" % (test_WER_mean_random*100, test_WER_std_random*100))
		print("FLOPs: %d" % (test_FLOPs_mean_random))
