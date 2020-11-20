import matplotlib.pyplot as plt
from glob import glob
import yaml
import os
import numpy as np
from pprint import pprint

import torch
import torchvision

nn_names = [
    'resnet',
    'densenet',
    'mobilenet',
    'squeezenet',
]

optim_names = [
    'Adam',
    'RAdam',
    'NVRMRadam',
    'LookAhead',
]

seeds = [
    1636004,
    150720,
    56889,
]

learning_rates = [
    '1e-3',
    '0.0003125',
    '3.125e-06',
]

metrics = ['grad_means', 'grad_variances', 'pert_acc', 'pert_loss', 'test_acc', 'test_loss', 'train_acc',
                'train_loss', 'train_times']

logdir = os.path.expanduser('~/pytorch-optim/')

print(logdir)

experiments = {}
experiments_averages = {}

for nn in nn_names:
    experiments[nn] = {}
    experiments_averages[nn] = {}

    for optim in optim_names:
        experiments[nn][optim] = {}
        experiments_averages[nn][optim] = {}

        for lr in learning_rates:

            if lr == '1e-3':
                log_files = glob(os.path.join(logdir, f'*_{nn}_{optim}'))  # because why not...
            else:
                log_files = glob(os.path.join(logdir, f'*_{nn}_{optim}_{lr}'))

            experiments[nn][optim][lr] = log_files

            if log_files:
                average_candidates = {k: [] for k in metrics}

                for fname in log_files:
                    with open(fname, 'r') as f:
                        content = yaml.load(f, Loader=yaml.Loader)

                        for k, v in content.items():
                            average_candidates[k].append(v)

                averages = {}

                for k, v in average_candidates.items():
                    average_candidates[k] = np.array(average_candidates[k])
                    averages[k] = np.mean(average_candidates[k], axis=0)

                experiments_averages[nn][optim][lr] = averages

# pprint(experiments_averages['mobilenet']['Adam'])

for nn in nn_names:
    for metric in metrics:

        for optim in optim_names:
            for lr in learning_rates:

                try:
                    plt.plot(experiments_averages[nn][optim][lr][metric], label=f'{optim} {lr}')
                except KeyError:
                    print(f'No data for {nn} {optim} {lr} {metric}')

        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.title(nn)

        plt.legend()
        # plt.show()
        plt.savefig(f'img/{nn}_{metric}.jpg')
        plt.clf()
