import os
import sys
import yaml
import pickle
import argparse
from numbers import Number
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from network.nets import *
from data.cmnist_data import *

class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(
        self,
        start_val=0,
        start_count=0,
        start_avg=0,
        start_sum=0
    ):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count
    def reset(self):
        """
            Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, num=1):
        """
            Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def binary_accuracy(output, target, threshold=0.5):
    with torch.no_grad():
        output_label = torch.zeros_like(output)
        output_label[ output > threshold ] = 1.
        batch_size = target.size(0)
        correct_results = torch.sum(output_label==target).item()
        return correct_results / batch_size


def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
    

def make_dirs(config):
    os.makedirs(config.reports_path, exist_ok=True)
    
    
def save_net(file_path, file_name, model, optimizer=None):
    """
        In this function, a model is saved.
        ------------------------------------------------
        Parameters:
            - file_path (str): saving path
            - file_name (str): saving name
            - model (torch.nn.Module)
            - optimizer (torch.optim)
    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_net(ckpt_path, model, optimizer=None):
    """
        Loading Network
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    if ((optimizer != None) & ("optimizer" in checkpoint.keys())):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer

def freeze_net(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
    

####################################training function#############################
def train_cee(config):
    """
        In this function, train a classifier on features learned in
        self-supervised manner.
    """
    ################ Dataloader ################
    envs = pickle.load(open("envs.pkl", "rb"))

    dataset = CMNIST(
        envs = [envs[0], envs[1]],
        device = config.device
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config.batch_size,
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = True,
        shuffle = True,
    )
    _, u_d_dim, u_y_dim, y_dim = dataset.get_dims()

    ################ Loss ################
    criterion = nn.BCELoss()
    ################ Model ################
    model = get_cee_model(
        config = config,
        u_y_dim = u_y_dim,
        u_d_dim = u_d_dim ,
        y_dim = y_dim
    )
    ################ Optimizer ################
    optimizer = optim.Adam(model.parameters(), lr = config.lr)
    ################ Learning Rate Scheduler ################
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor = 0.5,
        patience = 0,
        verbose = True
    )
    for epoch in range(1, config.epochs + 1):
        top1 = AverageMeter()
        losses = AverageMeter()
        train_report = pd.DataFrame({
            "mode": [],
            "epoch": [],
            "batch_id": [],
            "batch_size": [],
            "loss": [],
            "acc1": [],
            "lr": []
        })
        model.train()
        total_losss = 0
        with tqdm(data_loader) as t_data_loader:
            for it, (x, u_d , u_y, y, c_hat) in enumerate(t_data_loader):
                t_data_loader.set_description(
                    f"Training @ Epoch {epoch}")
                x = x.to(torch.float64).to(config.device)
                u_d = u_d.to(torch.float64).to(config.device)
                u_y = u_y.to(torch.float64).to(config.device)
                y = y.to(torch.float64).to(config.device)
                c_hat = c_hat.squeeze().to(torch.float64).to(config.device)
                c = model(x, u_d , u_y , y).squeeze()
                loss = criterion(c, c_hat) + config.regularization_coef * model.regularization_term()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc1 = binary_accuracy(c, c_hat)
                losses.update(loss.item(), x.size(0))
                top1.update(acc1, x.size(0))

                t_data_loader.set_postfix(
                    loss="{:.3f}".format(losses.avg),
                    acc1="{:.3f}".format(top1.avg),
                    lr="{:.5e}".format(optimizer.param_groups[0]["lr"])
                )

                batch_report = pd.DataFrame({
                        "mode": ["train"],
                        "epoch": [epoch],
                        "batch_id": [it],
                        "batch_size": [x.shape[0]],
                        "loss": [loss.item()],
                        "acc1": [acc1],
                        "lr": [optimizer.param_groups[0]["lr"]]
                    }
                )

                train_report = pd.concat([train_report, batch_report])

                total_losss += loss.item()
        # Saving the Modle
        save_net(
            file_path=config.reports_path,
            file_name="cee_{}.ckpt".format(epoch),
            model=model,
            optimizer=optimizer
        )

        # Saving Training Report
        train_report.to_csv(
            os.path.join(
                config.reports_path, f"train_{epoch}.csv"
            )
        )
        lr_scheduler.step(total_losss)
    return model, optimizer










