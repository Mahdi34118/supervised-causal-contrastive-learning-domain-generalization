import warnings
import os
import sys
import yaml
import pickle
import argparse
from numbers import Number
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import *
from data.cmnist_data import *
warnings.warn("deprecated", DeprecationWarning)
warnings.simplefilter("ignore")


# Load MNIST, make train/val splits, and shuffle train set examples
mnist = torchvision.datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])
rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())
# Build environment
envs = [
  make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2 , 0),
  make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1 , 1),
  make_environment(mnist_val[0], mnist_val[1], 0.9 , 2)
]
# Saving the envs list to a pickle file
with open('envs.pkl', 'wb') as f:
    pickle.dump(envs, f)

cee_training_yaml_file = "./cee_training.yaml"

# Read config file
with open(cee_training_yaml_file, 'r') as f:
  config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)
# Set device
config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Make nescessary directories
make_dirs(config)
# Set radnom seed
if config.random_seed:
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
# Train the CEE Model
model, optimizer = train_cee(config)
# Save the last layer weigths
last_layer_weights = model.corruption_classifier[0].weight
torch.save(last_layer_weights, os.path.join(config.reports_path, "last_layer_weights.pt"))

save_net(
    file_path=config.reports_path,
    file_name="cee_final.ckpt",
    model=model,
    optimizer=optimizer
)



