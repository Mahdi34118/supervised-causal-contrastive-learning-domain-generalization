import os
import sys
import yaml
import pickle
import argparse
from numbers import Number
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt 
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from matplotlib import pyplot as plt



class NoneActLayer(nn.Module):
    def __init__(self):
        super(NoneActLayer, self).__init__()
        pass
    def forward(self, x):
        return x

class XTanhActLayer(nn.Module):
    def __init__(self, coef_x=1., coef_tanh=1.):
        super(XTanhActLayer, self).__init__()
        self.coef_x = coef_x
        self.coef_tanh = coef_tanh
    def forward(self, x):
        return self.coef_tanh * x.tanh() + self.coef_x * x

    def __str__(self):
        return f"xtanh(coef_x={self.coef_x}, coef_tanh={self.coef_tanh})"
    def __repr__(self):
        return self.__str__()

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activations):
        super(MLP, self).__init__()

        # Dimensions
        assert isinstance(hidden_dims, list), f'Oops!! Wrong argument type for "hidden_dims": {type(hidden_dims)}. "hidden_dims" must be a list.'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Set activation functions
        if isinstance(activations, str):
            self.activations = [activations] * len(self.hidden_dims)
        elif isinstance(activations, list):
            assert len(activations)==len(hidden_dims)+1, f'Oops!! Wrong argument value for "activations". "activations" must have only one element more than "hidden_dims".'
            self.activations = activations
        else:
            raise ValueError(f'Oops!! Wrong Argument type for "activations": {activations}. "activations" must be one of this types: [str, list]')
        self.activations = MLP.get_activation_functions(self.activations)

        # Set the layers
        self.mlp = MLP.get_layers(
                        _input_dim = self.input_dim,
                        _hidden_dims = self.hidden_dims,
                        _output_dim = self.output_dim,
                        _acts = self.activations
                    )

    @staticmethod
    def get_activation_functions(acts):
        _activations = list()
        for act in acts:
            act = act.split("_")
            if act[0].lower() == "sigmoid":
                _activations.append(nn.Sigmoid())
            elif act[0].lower() == "xtanh":            # xtanh_0.1_10
                coef_x = 1 if len(act)==1 else float(act[1])
                coef_tanh = 1 if len(act)<3 else float(act[2])
                _activations.append(XTanhActLayer(coef_x=coef_x, coef_tanh=coef_tanh))
            elif act[0].lower() == "relu":
                _activations.append(nn.ReLU())
            elif act[0].lower() == "lrelu":
                _activations.append(nn.LeakyReLU())
            elif act[0].lower() == "none":
                _activations.append(NoneActLayer())
            else:
                raise ValueError(f'Oops!! Wrong argument value for "activations": {acts} -> {act}. "activations" must be one of this values: ["none", sigmoid", "xtanh", "relu", "lrelu"]')

        return _activations

    @staticmethod
    def get_layers(_input_dim, _hidden_dims, _output_dim, _acts):
        layers = list()

        layers_in = [_input_dim] + _hidden_dims
        layers_out = _hidden_dims + [_output_dim]

        for (_in, _out, _act) in zip(layers_in, layers_out, _acts):
            layers.append(
                nn.Linear(_in, _out)
            )
            layers.append(_act)

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)



class MLPDoubleHead(nn.Module):
    def __init__(self, input_dim, head1_dim, head2_dim, hidden_dims, hidden_activations, head1_activation, head2_activation) -> None:
        super().__init__()

        # Dimensions
        assert isinstance(hidden_dims, list), f'Oops!! Wrong argument type for "hidden_dims": {type(hidden_dims)}. "hidden_dims" must be a list.'
        self.input_dim = input_dim
        self.head1_dim = head1_dim
        self.head2_dim = head2_dim
        self.hidden_dims = hidden_dims

        self.base_mlp = None
        print(f"hidden_dims: {hidden_dims} {len(hidden_dims)}")
        if len(hidden_dims)!=0:
            self.base_mlp = MLP(
                input_dim = input_dim,
                hidden_dims = hidden_dims[:-1],
                output_dim = hidden_dims[-1],
                activations = hidden_activations
            )

        head1_activation = MLP.get_activation_functions([head1_activation])[0]
        head2_activation = MLP.get_activation_functions([head2_activation])[0]

        heads_in_dim = self.hidden_dims[-1] if len(self.hidden_dims)!=0 else self.input_dim
        self.head1 = nn.Sequential(
            nn.Linear(heads_in_dim, self.head1_dim),
            head1_activation
        )

        self.head2 = nn.Sequential(
            nn.Linear(heads_in_dim, head2_dim),
            head2_activation
        )

    def forward(self, x):
        rep = x
        if self.base_mlp != None:
            rep = self.base_mlp(x)

        out_head1 = self.head1(rep)
        out_head2 = self.head2(rep)

        return out_head1, out_head2


############# Nerwork_MODEL ###########################
class CMNISTEncoder(nn.Module):
    def __init__(self, latent_dim, device) -> None:
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        ).to(torch.float64).to(device)

    def forward(self, x):
        return self.encoder(x)
    def get_output_dim(self):
        _t = torch.rand((1, 2, 28, 28)).to(torch.float64).to(self.device)
        return self.forward(_t).shape[1]


class CEENet(nn.Module):
    def __init__(self, data_encoder, effect_extractor, env_latent_encoder, num_Zy ,  device) -> None:
        super().__init__()
        self.data_encoder = data_encoder
        self.effect_extractor = effect_extractor
        self.env_latent_encoder = env_latent_encoder
        self.latent_dim = self.data_encoder.get_output_dim()
        self.num_Zy = num_Zy
        self.num_Zd = self.latent_dim - self.num_Zy
        self.corruption_classifier = nn.Sequential(
            nn.Linear(self.num_Zy + self.num_Zd , 1),
            nn.Sigmoid()
        ).to(torch.float64).to(device)
    def forward(self, x, u_d , u_y, y):
        z = self.data_encoder(x)
        causal_effect = torch.zeros((z.shape[0], self.num_Zy+ self.num_Zd), dtype=torch.float64).to(x.device)
        ################ Phi(i, z_y_i, U) or Phi(z_y_i, U) ################
        for ix in range(self.num_Zy):
            _ix = ix * torch.ones((x.shape[0], 1), dtype=torch.float64).to(x.device)
            z_ix = z[:, ix].unsqueeze(dim=1)
            _in = torch.cat([z_ix, y], dim=1)                                                  # for module lists
            causal_effect[:, ix] = self.effect_extractor[ix](_in).squeeze()                    # ix th module of the module lists of phis
        ################ Q(Z_d, U) ################
        for ix in range(self.num_Zd):
            _ix = ix * torch.ones((x.shape[0], 1), dtype=torch.float64).to(x.device)
            z_ix = z[:, self.num_Zy + ix].unsqueeze(dim=1)
            _in = torch.cat([z_ix, u_d], dim=1)                                                 # for module lists
            causal_effect[:,self.num_Zy + ix] = self.env_latent_encoder[ix](_in).squeeze()
        ################ Corruption Classification ################
        corruption = self.corruption_classifier(causal_effect)
        return corruption
    def regularization_term(self):
        last_layer_weights = self.corruption_classifier[0].weight
        return torch.sum(last_layer_weights.abs())



################ integrated_model###############################
def get_cee_model(config, u_d_dim ,  u_y_dim, y_dim):
    num_Zy = config.num_Zy
    num_Zd = config.data_encoder_latent_dim - num_Zy
    data_encoder = CMNISTEncoder(config.data_encoder_latent_dim, config.device).to(torch.float64).to(config.device)
    causal_effect_extractor = nn.ModuleList(
      [
        MLP(
          # input_dim =  1 + u_y_dim + y_dim,
          input_dim =  1 + u_y_dim,
          output_dim = 1,
          hidden_dims = config.cee_hidden_dims,
          activations = config.cee_activations
        ).to(torch.float64).to(config.device)
        for _ in range(num_Zy)
      ]
      )

    env_latent_encoder = nn.ModuleList(
      [
        MLP(
          input_dim =  1 + u_d_dim ,
          output_dim = 1,
          hidden_dims = config.env_latent_encoder_hidden_dims,
          activations = config.env_latent_encoder_activations
        ).to(torch.float64).to(config.device)
        for _ in range(num_Zd)
      ]
      )
    cee = CEENet(
        data_encoder = data_encoder,
        effect_extractor = causal_effect_extractor,
        env_latent_encoder = env_latent_encoder,
        device = config.device,
        num_Zy = num_Zy
    ).to(torch.float64).to(config.device)
    return cee


