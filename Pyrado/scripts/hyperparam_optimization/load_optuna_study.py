"""
Load an Optuna study and print the best hyper-parameter set.
"""
import numpy as np
import optuna
import os
import os.path as osp
import pprint
from matplotlib.ticker import MaxNLocator

from pyrado.logger.experiment import ask_for_experiment
from matplotlib import pyplot as plt
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Find and load the Optuna data base
    study, study_name = None, None
    for file in os.listdir(ex_dir):
        if file.endswith('.db'):
            study_name = file[:-3]  # we named the file like the study
            storage = f'sqlite:////{osp.join(ex_dir, file)}'
            study = optuna.load_study(study_name, storage)
            break  # assuming there is only one database

    if study is None:
        print_cbt('No study found!', 'r', bright=True)

    # Extract the values of all trials (optuna was set to solve a minimization problem)
    values = np.array([t.value for t in study.trials])
    values = -1*values[values != np.array(None)]  # broken trials return None

    # Print the best parameter configuration
    pp = pprint.PrettyPrinter(indent=4)
    print_cbt(f'Best parameter set (trial_{study.best_trial.number}) from study {study_name} with average return '
              f'{-study.best_value}', 'g', bright=True)
    pp.pprint(study.best_params)

    # Plot a histogram
    fig, ax = plt.subplots(1, figsize=(8, 6))
    n, bins, patches = plt.hist(values, len(study.trials), density=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Histogram of the Returns')
    plt.xlabel('return')
    plt.ylabel('count')
    plt.grid(True)
    plt.show()
