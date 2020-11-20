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
    'resnet-small',
]

optim_names = [
    'Adam',
    'RAdam',
    'NVRMRadam',
    'LookAhead',
]

seeds = [
    '1636004',
    '150720',
    '56889',
]

learning_rates = [
    '1e-2',
    '1e-3',
    '1e-4',
]

variability = [
    '1.6e-2',
    '1.6e1',
    '1.6e0',
]

default_nvrmr = variability[0]

metrics = ['grad_means', 'grad_variances', 'pert_acc', 'pert_loss', 'test_acc', 'test_loss', 'train_acc',
                'train_loss', 'train_times']

metrics_translation = {
    'grad_means': 'Gradient Mean',
    'grad_variances': 'Gradient Variance',
    'pert_acc': 'Perturbed Accuracy',
    'pert_loss': 'Perturbed Loss',
    'test_acc': 'Test Accuracy',
    'test_loss': 'Test Loss',
    'train_acc': 'Train Accuracy',
    'train_loss': 'Train Loss',
    'train_times': 'Train Times',
}


logdir = os.path.expanduser('~/pytorch-optim/runs/')

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

            experiments[nn][optim][lr] = {}
            experiments_averages[nn][optim][lr] = {}

            if optim == 'NVRMRadam':

                for vb in variability:

                    log_files = glob(os.path.join(logdir, f'*_{nn}_{optim}_{lr}_{vb}'))
                    log_files += glob(os.path.join(logdir, f'*_{nn}_{optim}_{float(lr)}_{float(vb)}'))

                    experiments[nn][optim][lr][vb] = log_files

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

                        experiments_averages[nn][optim][lr][vb] = averages
            else:

                log_files = glob(os.path.join(logdir, f'*_{nn}_{optim}_{lr}'))
                log_files += glob(os.path.join(logdir, f'*_{nn}_{optim}_{float(lr)}'))

                experiments[nn][optim][lr][0] = log_files

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

                    experiments_averages[nn][optim][lr][0] = averages

# pprint(experiments_averages['mobilenet']['Adam'])

# optimizers showdown

for nn in nn_names:
    for metric in metrics:

        for optim in optim_names:
            for lr in learning_rates:

                try:
                    if optim == 'NVRMRadam':
                        plt.plot(experiments_averages[nn][optim][lr][default_nvrmr][metric], label=f'{optim} {lr}')
                    else:
                        plt.plot(experiments_averages[nn][optim][lr][0][metric], label=f'{optim} {lr}')
                except KeyError:
                    print(f'No data for {nn} {optim} {lr} {metric}')

        plt.xlabel('Epoch')
        plt.ylabel(metrics_translation[metric])
        plt.title(nn)

        plt.legend()
        # plt.show()
        plt.savefig(f'img/compare_opts/{nn}_{metric}.jpg')
        plt.clf()

# learning rates showdown

for nn in nn_names:
    for metric in metrics:
        for optim in optim_names:
            for lr in learning_rates:

                try:
                    if optim == 'NVRMRadam':
                        plt.plot(experiments_averages[nn][optim][lr][default_nvrmr][metric], label=f'{optim} {lr}')
                    else:
                        plt.plot(experiments_averages[nn][optim][lr][0][metric], label=f'{optim} {lr}')
                except KeyError:
                    print(f'No data for {nn} {optim} {lr} {metric}')

            plt.xlabel('Epoch')
            plt.ylabel(metrics_translation[metric])
            plt.title(nn)

            plt.legend()
            # plt.show()
            plt.savefig(f'img/compare_lr/{nn}_{metric}.jpg')
            plt.clf()

# variability showdown

for nn in nn_names:
    for metric in metrics:

        optim = 'NVRMRadam'

        for lr in learning_rates:
            for vb in variability:
                try:
                    plt.plot(experiments_averages[nn][optim][lr][vb][metric], label=f'{optim} lr={lr} var={vb}')
                except KeyError:
                    print(f'No data for {nn} {optim} {lr} {metric}')

        plt.xlabel('Epoch')
        plt.ylabel(metrics_translation[metric])
        plt.title(nn)

        plt.legend()
        # plt.show()
        plt.savefig(f'img/compare_vb/{nn}_{metric}.jpg')
        plt.clf()
