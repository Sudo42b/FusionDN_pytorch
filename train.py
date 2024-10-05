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
from utils.util import loss_plot, accuracy_plot

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



if __name__ == '__main__':
    epochs = 50
    lr = 1e-3
    batch_size = 128
    sample_size = 200
    num_task = 3
    
    # train_loader, test_loader = get_permute_mnist()
    # loss, acc, weight = standard_process(model=MODEL(),
    #                                     epochs=10,
    #                                     num_task=5,
    #                                     lr=1e-3,
    #                                     train_loader=train_loader,
    #                                     test_loader=test_loader)
    
    
    # loss_plot(loss, epochs)
    
    # accuracy_plot(acc, num_task, epochs)
    
    # loss_ewc, acc_ewc = ewc_process(epochs, 
    #                                 importance=1000, 
    #                                 # weight=weight,
    #                                 num_task=num_task,
    #                                 train_loader=train_loader,
    #                                 test_loader=test_loader)
    # loss_plot(loss_ewc, epochs)
    # accuracy_plot(acc_ewc, num_task=num_task, epochs=epochs)
    # plt.plot(acc[0], label="sgd")
    # plt.plot(acc_ewc[0], label="ewc")
    # plt.legend()
    