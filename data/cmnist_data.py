import os
import sys
import yaml
import pickle
import argparse
from numbers import Number
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

def plot_image(img_t, real_digit, corrupted_digit, corruption, corrupted_labels):
  img_t = torch.cat([img_t, torch.zeros((1, img_t.shape[1], img_t.shape[2]))])
  img_np = img_t.numpy()
  img_np = np.transpose(img_np, (1, 2, 0))
  img_np_zero = np.zeros
  plt.imshow(img_np)
  plt.axis('off')  # Turn off axis labels and ticks
  plt.title(f"Corruption: {corruption}\nCorrupted Labels:{corrupted_labels}\nReal Digit: {real_digit}\nCorrupted Digit: {corrupted_digit}")
  plt.show()


###################### DATA LOADER CLASS #########################
class CMNIST(Dataset):
    def __init__(self, envs, device) -> None:
        super().__init__()

        self.x = torch.cat([env["images"] for env in envs], dim=0)  # concatenate on 0 dim for two enviroments
        self.y = torch.cat([env["corrupted_labels"] for env in envs], dim=0)
        self.u_y = torch.cat([env["digit_labels"] for env in envs], dim=0)
        self.u_d = torch.cat([env["env_id"] for env in envs], dim=0)
        self.corruption = torch.cat([env["corrupted_mask"] for env in envs], dim=0)
        indecies = torch.arange(self.x.shape[0])
        indecies = indecies[torch.randperm(indecies.size()[0])]
        self.x = self.x[indecies]
        self.y = self.y[indecies]
        self.u_d = self.u_d[indecies]
        self.u_y = self.u_y[indecies]
        self.corruption = self.corruption[indecies]

    def get_dims(self):
        u_y_dim = 1
        u_d_dim = 1
        y_dim = 1
        return self.x.shape[1:], u_d_dim, u_y_dim, y_dim

    def __getitem__(self, index):
        _x = self.x[index]
        _u_d = torch.tensor([self.u_d[index]])
        _u_y = torch.tensor([self.u_y[index]])
        _y = torch.tensor([self.y[index]])
        _corruption = torch.tensor([self.corruption[index]])
        return _x, _u_d , _u_y , _y, _corruption

    def __len__(self):
        return self.x.shape[0]



###################### train_data_maker #########################

def make_environment(images, digit_labels, e , env_id):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1

  # Assign a binary label based on the digit; flip label with probability 0.25
  real_digit_labels = digit_labels.clone()      # copy the tensor
  correct_labels = (digit_labels < 5).float()
  corrupted_mask = torch_bernoulli(0.25, len(correct_labels))
  corrupted_labels = torch_xor(correct_labels, corrupted_mask)
  # Make digits and corrupted labels consistent
  label_0_corruption_mask = (correct_labels == 0) & corrupted_mask.to(torch.bool)
  label_1_corruption_mask = (correct_labels == 1) & corrupted_mask.to(torch.bool)
  n_label_0_corruption_mask = torch.sum(label_0_corruption_mask.to(torch.int)).item()
  n_label_1_corruption_mask = torch.sum(label_1_corruption_mask.to(torch.int)).item()
  label_0_non_cosistence_labels = np.random.choice([0, 1, 2, 3, 4], size=n_label_0_corruption_mask)
  label_1_non_cosistence_labels = np.random.choice([5, 6, 7, 8, 9], size=n_label_1_corruption_mask)
  digit_labels[label_0_corruption_mask] = torch.tensor(label_0_non_cosistence_labels)
  digit_labels[label_1_corruption_mask] = torch.tensor(label_1_non_cosistence_labels)
  env_ind = torch.tensor([env_id] * corrupted_mask.shape[0])
  corrupted_env_ind = env_ind.clone()
  corrupted_env_ind[corrupted_mask.bool()] = 1 - env_ind[corrupted_mask.bool()]

  # Assign a color based on the label; flip the color with probability e
  colors = torch_xor(corrupted_labels, torch_bernoulli(e, len(corrupted_labels)))

  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images], dim=1)
  images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
  return {
    'images': (images.float() / 255.),
    'real_digit_labels': real_digit_labels,
    'digit_labels': digit_labels,
    'corrupted_labels': corrupted_labels,
    'corrupted_mask': corrupted_mask * 1. ,
    'env_id': corrupted_env_ind
  }











