# This is a sample Python script.
import torch
import torchvision
import time
#from multiprocessing import Pool, Process
import multiprocessing
from multiprocessing import pool

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.densenet import densenet121
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.squeezenet import squeezenet1_1
from utils import *
from math import sqrt
from typing import Final
from tqdm import tqdm


import yaml
import json
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST, ImageFolder


def make_one_hot(target: int, num_classes=10):
    # kinda slow but this bullshit pytorch wants per element function
    target = torch.ones(size=(10,)) * target
    target = (target == torch.arange(num_classes))
    return target.type(torch.FloatTensor)


def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2839, 0.2804, 0.3028), std=(0.4625, 0.4579, 0.4299))  # normalising to imagenette2 statistics
    ])
    return transform(image)


def perturb_transform(image, perturb_alpha=0.25):
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2839, 0.2804, 0.3028), std=(0.4625, 0.4579, 0.4299))  # normalising to imagenette2 statistics
    ])
    image = transform(image)
    image = image + perturb_alpha * torch.randn_like(image)
    return image


class ImageNetteDataset(ImageFolder):
    def __init__(self, root, transform, target_transform):
        super(ImageNetteDataset, self).__init__(root=root, transform=transform, target_transform=target_transform)


class Optimizer(object):
    def __init__(self, params, **kwargs):
        self.params = params
        self.options = kwargs
        self.lr = self.options['lr']
        self.t_step = 1.

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = p.grad.data.zero_()

    def step(self):
        pass

    @torch.no_grad()
    def apply_schedule(self, schedule_fn, epoch, epoch_stamps):
        if epoch in epoch_stamps:
            self.lr = schedule_fn(self.lr)


class SGD(Optimizer):
    def __init__(self, params, **kwargs):
        super(SGD, self).__init__(params, **kwargs)

    def update_state(self):
        pass

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= p.grad.data * self.lr


class Adam(Optimizer):

    def __init__(self, params, **kwargs):
        super(Adam, self).__init__(params, **kwargs)
        self.beta1 = self.options['beta1']
        self.beta2 = self.options['beta2']
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}

    def reset_state(self):
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.t_step = 1

    @torch.no_grad()
    def step(self):
        # compute bias correction
        bias_correction1 = 1 - self.beta1 ** self.t_step
        bias_correction2 = 1 - self.beta2 ** self.t_step

        for p in self.params:
            if p.grad is not None:
                # compute exponential moving average
                self.m[p] = self.m[p] * self.beta1 + p.grad.data * (1 - self.beta1)

                # compute exponential squared moving average
                self.v[p] = self.v[p] * self.beta2 + p.grad.data ** 2 * (1 - self.beta2)

                # compute the adaptive learning rate
                l_t = self.v[p].sqrt() / (sqrt(bias_correction2) + 1e-7) + 1e-7

                # compute step_size -> includes bias-correction for 1st moment EMA
                step_size = self.lr / bias_correction1

                # apply return update
                p.data -= self.m[p] / l_t * step_size

        # update current optimization step
        self.t_step = self.t_step + 1


class NVMRAdam(Optimizer):

    def __init__(self, params, **kwargs):
        super(NVMRAdam, self).__init__(params, **kwargs)
        self.beta1 = self.options['beta1']
        self.beta2 = self.options['beta2']
        self.variability = self.options['variability']
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.n = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}


    def reset_state(self):
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.n = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.t_step = 1

    @torch.no_grad()
    def step(self):
        # compute bias correction
        bias_correction1 = 1 - self.beta1 ** self.t_step
        bias_correction2 = 1 - self.beta2 ** self.t_step

        for p in self.params:
            if p.grad is not None:
                # compute exponential moving average
                self.m[p] = self.m[p] * self.beta1 + p.grad.data * (1 - self.beta1)

                # compute exponential squared moving average
                self.v[p] = self.v[p] * self.beta2 + p.grad.data ** 2 * (1 - self.beta2)

                # compute the adaptive learning rate
                l_t = self.v[p].sqrt() / (sqrt(bias_correction2) + 1e-7) + 1e-7

                # compute step_size -> includes bias-correction for 1st moment EMA
                step_size = self.lr / bias_correction1

                # apply return update
                p.data -= self.m[p] / l_t * step_size

                # NVRM update
                noise = torch.normal(torch.zeros_like(p.data), self.variability)
                p.data += noise - self.n[p]
                self.n[p].copy_(noise)

        # update current optimization step
        self.t_step = self.t_step + 1


class RAdam(Optimizer):

    def __init__(self, params, **kwargs):
        super(RAdam, self).__init__(params, **kwargs)
        self.beta1 = self.options['beta1']
        self.beta2 = self.options['beta2']
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.rho_max = 2 / (1 - self.beta2) - 1

    def reset_state(self):
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.t_step = 1

    @torch.no_grad()
    def step(self):

        for p in self.params:
            if p.grad is not None:
                # compute exponential moving average
                self.m[p] = self.m[p] * self.beta1 + p.grad.data * (1 - self.beta1)

                # compute exponential squared moving average
                self.v[p] = self.v[p] * (1 / self.beta2) + p.grad.data ** 2 * (1 - self.beta2)

                # update beta2
                beta2_t = self.beta2 ** self.t_step

                # compute rho
                rho = self.rho_max - 2 * self.t_step * beta2_t / (1 - beta2_t)

                # compute step_size -> includes bias-correction for 1st moment EMA
                step_size = self.lr / (1 - self.beta1 ** self.t_step)

                # if approximated value goes beyond threshold
                if rho > 5:
                    # compute variance rectification
                    r = sqrt(
                        (rho - 4) * (rho - 2) * self.rho_max / ((self.rho_max - 4) * (self.rho_max - 2) * rho)
                    )

                    # compute adaptive learning rate
                    l_t = sqrt(1 - beta2_t) / (self.v[p].sqrt() + 1e-7) + 1e-7

                    # add variance rectification to step size
                    step_size = step_size * r
                else:
                    # adaptive learning rate is 1.
                    l_t = 1.

                # update weight
                p.data -= self.m[p] * l_t * step_size

        # update current optimization step
        self.t_step = self.t_step + 1


class NVRMRadam(Optimizer):

    def __init__(self, params, **kwargs):
        super(NVRMRadam, self).__init__(params, **kwargs)
        self.beta1 = self.options['beta1']
        self.beta2 = self.options['beta2']
        self.variability = self.options['variability']
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.n = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.rho_max = 2 / (1 - self.beta2) - 1

    def reset_state(self):
        self.m = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.v = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.n = {p: torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params}
        self.t_step = 1

    @torch.no_grad()
    def step(self):

        for p in self.params:
            if p.grad is not None:
                # compute exponential moving average
                self.m[p] = self.m[p] * self.beta1 + p.grad.data * (1 - self.beta1)

                # compute exponential squared moving average
                self.v[p] = self.v[p] * (1 / self.beta2) + p.grad.data ** 2 * (1 - self.beta2)

                # update beta2
                beta2_t = self.beta2 ** self.t_step

                # compute rho
                rho = self.rho_max - 2 * self.t_step * beta2_t / (1 - beta2_t)

                # compute step_size -> includes bias-correction for 1st moment EMA
                step_size = self.lr / (1 - self.beta1 ** self.t_step)

                # if approximated value goes beyond threshold
                if rho > 5:
                    # compute variance rectification
                    r = sqrt(
                        (rho - 4) * (rho - 2) * self.rho_max / ((self.rho_max - 4) * (self.rho_max - 2) * rho)
                    )

                    # compute adaptive learning rate
                    l_t = sqrt(1 - beta2_t) / (self.v[p].sqrt() + 1e-7) + 1e-7

                    # add variance rectification to step size
                    step_size = step_size * r
                else:
                    # adaptive learning rate is 1.
                    l_t = 1.

                # update weight
                p.data -= self.m[p] * l_t * step_size

                # NVRM update
                noise = torch.normal(torch.zeros_like(p.data), self.variability)
                p.data += noise - self.n[p]
                self.n[p].copy_(noise)

        # update current optimization step
        self.t_step = self.t_step + 1


class LookAhead(Optimizer):
    def __init__(self, params, **kwargs):
        super(LookAhead, self).__init__(params, **kwargs)
        self._slow_params = {p: torch.zeros_like(p.data, memory_format=torch.preserve_format).copy_(p.data) for p in params}
        self.optimizer = self.options['optimizer']
        self.alpha = self.options['alpha']

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        if (self.optimizer.t_step - 1) % self.options['la_steps'] == 0:
            for p in self.optimizer.params:
                res = self.alpha * p.data + self._slow_params[p] * (1 - self.alpha)
                self._slow_params[p].data.copy_(res)
                p.data.copy_(res)
            self.optimizer.reset_state()

    @torch.no_grad()
    def apply_schedule(self, schedule_fn, epoch, epoch_stamps):
        self.optimizer.apply_schedule(schedule_fn, epoch, epoch_stamps)


def train_model(model, optimizer, loss_fn, loader: DataLoader):
    acc = 0.
    total_loss = 0.

    device = next(model.parameters()).device

    for idx, sample in enumerate(loader):
        optimizer.zero_grad()

        X, y = sample
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)

        matches = torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)
        matches = matches.type(torch.FloatTensor)
        acc += torch.sum(matches)

        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
        loss.backward(retain_graph=True)

        optimizer.step()

    acc /= len(loader.dataset)
    total_loss /= idx
    return total_loss, acc


def test_model(model, loss_fn, loader: DataLoader):
    acc = 0.
    total_loss = 0.

    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, sample in enumerate(loader):
            X, y = sample
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)

            matches = torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)
            matches = matches.type(torch.FloatTensor)
            acc += torch.sum(matches).item()

            loss = loss_fn(y_hat, y)
            total_loss += loss.item()

    acc /= len(loader.dataset)
    total_loss /= idx
    return total_loss, acc


def grad_mean(params):
    acc = 0.
    with torch.no_grad():
        for p in params:
            acc += torch.mean(p.grad.data.cpu()).item()
        return acc / len(params)


def grad_var(params):
    acc = 0.
    with torch.no_grad():
        for p in params:
            acc += torch.var(p.grad.data.cpu()).item()
        return acc / len(params)


def run_experiment(model, optimizer, loaders, loss_fn, num_epochs):
    test_acc, train_acc, train_loss, test_loss = [], [], [], []
    pert_acc, pert_loss = [], []
    grad_variances = []
    grad_means = []

    train_times, test_times = [], []
    train_loader, test_loader, perturb_loader = loaders

    for e in range(num_epochs):
        start_train = time.time()
        train_metrics = train_model(model, optimizer, loss_fn, train_loader)
        train_time = time.time() - start_train
        train_times.append(train_time)

        grad_variances.append(grad_var(list(model.parameters())))
        grad_means.append(grad_mean(list(model.parameters())))

        test_metrics = test_model(model, loss_fn, test_loader)
        pert_metrics = test_model(model, loss_fn, perturb_loader)
        optimizer.apply_schedule(schedule_fn=lambda x: x * 0.5, epoch=e + 1, epoch_stamps=[10, 20, 30, 40, 45])

        train_loss.append(train_metrics[0])
        train_acc.append(train_metrics[1])

        test_loss.append(test_metrics[0])
        test_acc.append(test_metrics[1])

        pert_loss.append(pert_metrics[0])
        pert_acc.append(pert_metrics[1])

        print('At epoch:{} --- Train accuracy: {}, Train loss: {}'.format(e, train_acc[-1], train_loss[-1]))
        print('At epoch:{} --- Test accuracy: {}, Test loss: {}'.format(e, test_acc[-1], test_loss[-1]))

    return {'test_acc': test_acc, 'train_acc': train_acc, 'test_loss': test_loss, 'train_loss': train_loss,
            'grad_variances': grad_variances, 'pert_acc': pert_acc, 'pert_loss': pert_loss,
            'grad_means': grad_means, 'train_times': train_times}


def run_across_seeds(config):
    # torch.backends.cudnn.deterministic = True

    torch.manual_seed(0)

    device = torch.device('cuda:{}'.format(int(config['device'])))
    model_type = config['model_type']
    opt_type = config['opt_type']

    batch_size = 64
    seeds = [150720, 56889, 1636004]

    target_transform = make_one_hot
    test_transform = image_transform
    test_p_transform = perturb_transform

    train_set = ImageNetteDataset('imagenette2-320/train', transform=image_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, num_workers=2, pin_memory=True, shuffle=True,
                              batch_size=batch_size)

    test_set = ImageNetteDataset('imagenette2-320/val', transform=test_transform, target_transform=target_transform)
    test_loader = DataLoader(test_set, num_workers=1, pin_memory=True, shuffle=True, batch_size=batch_size)

    perturb_set = ImageNetteDataset('imagenette2-320/val', transform=test_p_transform, target_transform=target_transform)
    perturb_loader = DataLoader(perturb_set, num_workers=1, pin_memory=True, shuffle=True, batch_size=batch_size)

    loaders = (train_loader, test_loader, perturb_loader)

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = get_model_constructor(model_type)(pretrained=False, progress=False, num_classes=10)
        model = model.to(device)
        model = torch.jit.trace(model, example_inputs=torch.ones(size=(128, 3, 256, 256), device=device))
        opt = get_opt_constructor(opt_type, params=list(model.parameters()), config=config)()
        results[seed] = run_experiment(model, opt, loaders, loss_fn, num_epochs=50)

        path_string = 'runs/' + '_'.join([str(seed), model_type, opt_type, str(opt.options['lr'])])
        if opt_type == 'NVRMAdam' or opt_type == 'NVRMRadam':
            path_string = path_string + '_' + str(config['variability'])

        with open(path_string, 'w') as f:
            yaml.dump(results[seed], f)
    return results


def get_model_constructor(model_type):
    if model_type == 'resnet':
        return resnet101
    elif model_type == 'densenet':
        return densenet121
    elif model_type == 'mobilenet':
        return mobilenet_v2
    elif model_type == 'squeezenet':
        return squeezenet1_1
    elif model_type == 'resnet-small':
        return resnet18


def get_opt_constructor(opt_type, params, config):
    if opt_type == 'Adam':
        return partial(Adam, params, lr=config['lr'], beta1=0.9, beta2=0.99)
    elif opt_type == 'RAdam':
        return partial(RAdam, params, lr=config['lr'], beta1=0.9, beta2=0.99)
    elif opt_type == 'NVRMAdam':
        return partial(NVMRAdam, params, lr=config['lr'], beta1=0.9, beta2=0.99, variability=config['variability'])
    elif opt_type == 'NVRMRadam':
        return partial(NVRMRadam, params, lr=config['lr'], beta1=0.9, beta2=0.99, variability=config['variability'])
    elif opt_type == 'LookAheadRAdam':
        slow_opt = RAdam(params=params, lr=config['lr'], beta1=0.9, beta2=0.99)
        return partial(LookAhead, params=params, alpha=0.5, la_steps=6, optimizer=slow_opt, lr=config['lr'])
    elif opt_type == 'LookAheadAdam':
        slow_opt = Adam(params=params, lr=config['lr'], beta1=0.9, beta2=0.99)
        return partial(LookAhead, params=params, alpha=0.5, la_steps=6, optimizer=slow_opt, lr=config['lr'])


def generate_experiment(model_type, opt_type, opt_lr, variability, device):
    return {
        'model_type': model_type,
        'opt_type': opt_type,
        'lr': opt_lr,
        'variability': variability,
        'device': device
    }


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    todo_experiments = [
        generate_experiment('resnet-small', 'LookAheadAdam', 1e-3, 1.6e-2, 0),
        generate_experiment('resnet-small', 'LookAheadAdam', 1e-2, 1.6e-2, 0),
        generate_experiment('resnet-small', 'LookAheadAdam', 1e-1, 1.6e-2, 0),
        generate_experiment('resnet-small', 'LookAheadRAdam', 1e-3, 1.6e-2, 0),
        generate_experiment('resnet-small', 'LookAheadRAdam', 1e-2, 1.6e-2, 0),
        generate_experiment('resnet-small', 'LookAheadRAdam', 1e-1, 1.6e-2, 3),
        generate_experiment('resnet-small', 'NVRMAdam', 1e-3, 1.6e-2, 2),
        generate_experiment('resnet-small', 'NVRMAdam', 1e-2, 1.6e-2, 2),
        generate_experiment('resnet-small', 'NVRMAdam', 1e-1, 1.6e-2, 2),
        generate_experiment('resnet-small', 'NVRMAdam', 1e-3, 1.6e1, 2),
        generate_experiment('resnet-small', 'NVRMAdam', 1e-3, 1.6e0, 2),
        generate_experiment('resnet-small', 'NVRMAdam', 1e-3, 1.6e-2, 3),
        generate_experiment('resnet-small', 'NVRMRadam', 1e-3, 1.6e1, 3),
        generate_experiment('resnet-small', 'NVRMRadam', 1e-3, 1.6e0, 3),
        generate_experiment('resnet-small', 'NVRMRadam', 1e-3, 1.6e-2, 3),
    ]

    p = NestablePool(len(todo_experiments))
    results = list(tqdm(p.imap(run_across_seeds, todo_experiments), total=len(todo_experiments)))
