from subprocess import call
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

experiments_folder = "experiments"
base_config_name = "mini-librispeech" #"timit"
base_config_path = os.path.join(experiments_folder, base_config_name + ".cfg")

#for seed in [1,2,3,4,5]:
	# make big-only config
	#call("python main.py --train --config_path=\"experiments/timit_big_only.cfg\"", shell=True)
	# make small-only config
	#call("python main.py --train --config_path=\"experiments/timit_small_only.cfg\"", shell=True)

class Experiment:
	def __init__(self, use_AR_features, surprisal_triggered_sampling_during_training, surprisal_triggered_sampling_during_testing, num_seeds, experiments_folder, base_config_name, base_config_path, big_only=False, small_only=False):
		self.use_AR_features = use_AR_features
		self.surprisal_triggered_sampling_during_training = surprisal_triggered_sampling_during_training
		self.surprisal_triggered_sampling_during_testing = surprisal_triggered_sampling_during_testing
		self.num_seeds = num_seeds
		self.experiments_folder = experiments_folder
		self.base_config_name = base_config_name
		self.base_config_path = base_config_path
		self.big_only = big_only
		self.small_only = small_only
		if self.big_only:
			self.surprisal_triggered_sampling_during_training = False
			self.surprisal_triggered_sampling_during_testing = False
			self.probability_of_sampling_big_during_training=1.0
			self.probability_of_sampling_big_during_testing=1.0
		if self.small_only:
			self.surprisal_triggered_sampling_during_training = False
			self.surprisal_triggered_sampling_during_testing = False
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
			self.name = self.base_config_name + "_%d_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training), int(self.surprisal_triggered_sampling_during_testing))
		for seed in range(1,self.num_seeds):
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
			surprisal_triggered_sampling_during_testing_line = ["sample_based_on_surprisal_during_testing=" in line for line in lines].index(True)
			lines[surprisal_triggered_sampling_during_testing_line]="sample_based_on_surprisal_during_testing="+str(self.surprisal_triggered_sampling_during_testing)+"\n"	
			probability_of_sampling_big_during_training_line = ["probability_of_sampling_big_during_training=" in line for line in lines].index(True)
			lines[probability_of_sampling_big_during_training_line] = "probability_of_sampling_big_during_training="+str(self.probability_of_sampling_big_during_training)+"\n"
			probability_of_sampling_big_during_testing_line = ["probability_of_sampling_big_during_testing=" in line for line in lines].index(True)
			lines[probability_of_sampling_big_during_testing_line] = "probability_of_sampling_big_during_testing="+str(self.probability_of_sampling_big_during_testing)+"\n"

			with open(experiment_config_path, "w") as f:
				f.writelines(lines)

			# Run the experiment
			call("python main.py --train --config_path=\""+ experiment_config_path +"\"", shell=True)

	def get_results(self, column, set):
		#self.name = self.base_config_name + "_%d_%d_%d" % (int(self.use_AR_features), int(self.surprisal_triggered_sampling_during_training), int(self.surprisal_triggered_sampling_during_testing))

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

# run experiments
use_AR_features = True; surprisal_triggered_sampling_during_training=False; surprisal_triggered_sampling_during_testing=False
# big only
experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, surprisal_triggered_sampling_during_testing, num_seeds, experiments_folder, base_config_name, base_config_path, big_only=True)
experiment.run()
experiments.append(experiment)
# small only
experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, surprisal_triggered_sampling_during_testing, num_seeds, experiments_folder, base_config_name, base_config_path, small_only=True)
experiment.run()
experiments.append(experiment)
# other
for use_AR_features in [True]: #[False, True]:
	for surprisal_triggered_sampling_during_training in [False, True]:
		for surprisal_triggered_sampling_during_testing in [False, True]:
			experiment = Experiment(use_AR_features, surprisal_triggered_sampling_during_training, surprisal_triggered_sampling_during_testing, num_seeds, experiments_folder, base_config_name, base_config_path)
			experiment.run()
			experiments.append(experiment)

# plot results
for experiment in experiments:
	valid_loss_mean, valid_loss_std = experiment.get_results(column="WER", set="valid")
	num_epochs = len(valid_loss_mean)
	color = "green" if experiment.surprisal_triggered_sampling_during_testing else "red"
	linestyle = "-" if experiment.surprisal_triggered_sampling_during_training else "--"
	color = "blue" if experiment.small_only else color
	color = "black" if experiment.big_only else color
	plt.plot(np.arange(0,num_epochs), valid_loss_mean, linestyle=linestyle, color=color, label=experiment.name)
	upper = valid_loss_mean + valid_loss_std
	lower = valid_loss_mean - valid_loss_std
	plt.fill_between(np.arange(0,num_epochs), lower, upper, facecolor=color, alpha=0.5)
plt.legend()
plt.show()

# print test results
for experiment in experiments:
	test_loss_mean, test_loss_std = experiment.get_results(column="loss", set="test")
	test_WER_mean, test_WER_std = experiment.get_results(column="WER", set="test")
	print(experiment.name)
	print("loss: %f +/- %f" % (test_loss_mean, test_loss_std))
	print("WER: %f +/- %f" % (test_WER_mean, test_WER_std))
