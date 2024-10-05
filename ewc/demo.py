import matplotlib.pyplot as plt

plt.style.use("seaborn-white")

import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

# from data import PermutedMNIST
from ewc.utils import EWC, ewc_train, normal_train, test
from models import MODEL
from datasets import get_dataloader
from utils.util import AverageMeter, ProgressMeter, reproducibility, accuracy

from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict

def standard_process(model:Module, 
                    epochs:int, 
                    num_task: int,
                    lr: float,
                    train_loader: Dict[DataLoader],
                    test_loader: Dict[DataLoader],
                    use_cuda:bool=True, 
                    weight:Dict=True)->Tuple[dict, dict, dict]:
    
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    loss, acc = {}, {}
    for task in range(num_task):
        loss[task] = []
        acc[task] = []
        
        
        for _ in tqdm(range(epochs)):
            loss[task].append(normal_train(model, optimizer, train_loader[task]))
            for sub_task in range(task + 1):
                acc[sub_task].append(test(model, test_loader[sub_task]))
        if task == 0 and weight:
            weight = model.state_dict()
    return loss, acc, weight



def ewc_process(model:Module, 
                epochs:int, 
                num_task: int,
                train_loader: Dict[DataLoader],
                test_loader: Dict[DataLoader],
                importance:int, 
                lr: float = 1e-3,
                sample_size:int=200,
                use_cuda:bool=True, 
                weight:dict=None)->Tuple[dict, dict]:
    
    def get_sample(dataset, sample_size):
        sample_idx = random.sample(range(len(dataset)), sample_size)
        return [img for img in dataset[sample_idx]]
    
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    optimizer = optim.SGD(params=model.parameters(), lr=lr)

    loss, acc, ewc = {}, {}, {}
    for task in range(num_task):
        loss[task] = []
        acc[task] = []

        if task == 0:
            if weight:
                model.load_state_dict(weight)
            else:
                for _ in tqdm(range(epochs)):
                    loss[task].append(normal_train(model, optimizer, train_loader[task]))
                    acc[task].append(test(model, test_loader[task]))
        else:
            old_tasks = []
            for sub_task in range(task):
                old_tasks = old_tasks + get_sample(train_loader[sub_task].dataset, sample_size)
            old_tasks = random.sample(old_tasks, k=sample_size)
            for _ in tqdm(range(epochs)):
                loss[task].append(ewc_train(model, optimizer, train_loader[task], EWC(model, old_tasks), importance))
                for sub_task in range(task + 1):
                    acc[sub_task].append(test(model, test_loader[sub_task]))

    return loss, acc


def loss_plot(x:dict, epochs:int):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x:dict, num_task:int, epochs:int):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)
    

if __name__ == '__main__':
    epochs = 50
    lr = 1e-3
    batch_size = 128
    sample_size = 200
    hidden_size = 200
    num_task = 3
    from torchvision import datasets
    class PermutedMNIST(datasets.MNIST):
        def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
            super(PermutedMNIST, self).__init__(root, train, download=True)
            assert len(permute_idx) == 28 * 28
            if self.train:
                self.train_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                            for img in self.train_data])
            else:
                self.test_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                            for img in self.test_data])
        def __getitem__(self, index):
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]
            return img, target
        
    def get_permute_mnist():
        train_loader = {}
        test_loader = {}
        idx = list(range(28 * 28))
        for i in range(num_task):
            train_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=True, permute_idx=idx),
                                                        batch_size=batch_size,
                                                        num_workers=4)
            test_loader[i] = torch.utils.data.DataLoader(PermutedMNIST(train=False, permute_idx=idx),
                                                        batch_size=batch_size)
            random.shuffle(idx)
        return train_loader, test_loader


    train_loader, test_loader = get_permute_mnist()
    loss, acc, weight = standard_process(model=MODEL(),
                                        epochs=10,
                                        num_task=5,
                                        lr=1e-3,
                                        train_loader=train_loader,
                                        test_loader=test_loader)
    
    
    loss_plot(loss, epochs)
    
    accuracy_plot(acc, num_task, epochs)
    
    loss_ewc, acc_ewc = ewc_process(epochs, 
                                    importance=1000, 
                                    # weight=weight,
                                    num_task=num_task,
                                    train_loader=train_loader,
                                    test_loader=test_loader)
    loss_plot(loss_ewc, epochs)
    accuracy_plot(acc_ewc, num_task=num_task, epochs=epochs)
    plt.plot(acc[0], label="sgd")
    plt.plot(acc_ewc[0], label="ewc")
    plt.legend()
    