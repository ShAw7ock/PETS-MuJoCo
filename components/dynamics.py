import copy
import torch
import torch.nn.functional as F
from torch import nn as nn

from torch.distributions import Normal


def linear(x):
    return x


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size, non_linearity):
        super(EnsembleDenseLayer, self).__init__()

        weights = torch.zeros(ensemble_size, n_in, n_out).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()

        for weight in weights:
            weight.transpose_(1, 0)

            if non_linearity == 'leaky_relu':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'swish':
                nn.init.xavier_uniform_(weight)
            elif non_linearity == 'tanh':
                nn.init.kaiming_normal_(weight)
            elif non_linearity == 'linear':
                nn.init.xavier_normal_(weight)

            weight.transpose_(1, 0)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        if non_linearity == 'leaky_relu':
            self.non_linearity = F.leaky_relu
        elif non_linearity == 'swish':
            self.non_linearity = swish
        elif non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif non_linearity == 'linear':
            self.non_linearity = linear

    def forward(self, inp):
        return self.non_linearity(torch.baddbmm(self.biases, inp, self.weights))


class NormalDynamicModel(nn.Module):
    min_log_var = -5
    max_log_var = -1

    def __init__(self, d_state, d_action, n_units, n_layers, ensemble_size, activation, device):
        assert n_layers >= 2

        super(NormalDynamicModel, self).__init__()

        layers = []
        for lyr_idx in range(n_layers + 1):
            if lyr_idx == 0:
                lyr = EnsembleDenseLayer(d_action + d_state, n_units, ensemble_size, non_linearity=activation)
            elif 0 < lyr_idx < n_layers:
                lyr = EnsembleDenseLayer(n_units, n_units, ensemble_size, non_linearity=activation)
            else:  # lyr_idx == n_layers:
                lyr = EnsembleDenseLayer(n_units, d_state + d_state, ensemble_size, non_linearity='linear')
            layers.append(lyr)

        self.layers = nn.Sequential(*layers)

        self.to(device)

        self.normalizer = None

        self.d_action = d_action
        self.d_state = d_state
        self.n_hidden = n_units
        self.n_layers = n_layers
        self.ensemble_size = ensemble_size
        self.device = device

    def setup_normalizer(self, normalizer):
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is not None:
            state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, var):
        # denormalize to return in raw state space
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            var = self.normalizer.denormalize_state_delta_vars(var)
        return delta_mean, var

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.layers(inp)
        delta_mean, log_var = torch.split(op, self.d_state, dim=2)

        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(log_var)

        return delta_mean, log_var.exp()

    def forward(self, states, actions):
        """
        predict next state mean and variance.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])

        Returns:
            next state means (torch Tensor[ensemble_size, batch size, dim_state])
            next state variances (torch Tensor[ensemble_size, batch size, dim_state])
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        normalized_states, normalized_actions = self._pre_process_model_inputs(states, actions)
        normalized_delta_mean, normalized_var = self._propagate_network(normalized_states, normalized_actions)
        delta_mean, var = self._post_process_model_outputs(normalized_delta_mean, normalized_var)
        next_state_mean = delta_mean + states
        return next_state_mean, var

    def forward_all(self, states, actions):
        """
        predict next state mean and variance of a batch of states and actions for all models in the ensemble.
        takes in raw states and actions and internally normalizes it.

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])

        Returns:
            next state means (torch Tensor[batch size, ensemble_size, dim_state])
            next state variances (torch Tensor[batch size, ensemble_size, dim_state])
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        states = states.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        next_state_means, next_state_vars = self(states, actions)
        return next_state_means.transpose(0, 1), next_state_vars.transpose(0, 1)

    def _random(self, states, actions):
        """ Returns a distribution for a single model in the ensemble (selected at random) """
        batch_size = states.shape[0]
        # Get next state distribution for all components in the ensemble
        next_state_means, next_state_vars = self.forward_all(states, actions)  # shape: (batch_size, ensemble_size, d_state)

        i = torch.arange(batch_size, device=self.device)
        j = torch.randint(self.ensemble_size, size=(batch_size,), device=self.device)
        mean = next_state_means[i, j]
        var = next_state_vars[i, j]

        return Normal(mean, var.sqrt())

    def _ensemble(self, states, actions):
        """ Create a single Gaussian out of Gaussian ensemble """
        # Get next state distribution for all components in the ensemble
        next_state_means, next_state_vars = self.forward_all(states, actions)  # shape: (batch_size, ensemble_size, d_state)
        next_state_means, next_state_vars = next_state_means.double(), next_state_vars.double()  # to prevent numerical errors (large means, small vars)

        mean = torch.mean(next_state_means, dim=1)  # shape: (batch_size, d_state)
        mean_var = torch.mean(next_state_vars, dim=1)  # shape: (batch_size, d_state)
        var = torch.mean(next_state_means ** 2, dim=1) - mean ** 2 + mean_var  # shape: (batch_size, d_state) 

        # A safety bound to prevent some unexpected numerical issues. The variance cannot be smaller then mean_var since
        # the sum of the other terms needs to be always positive (convexity)
        var = torch.max(var, mean_var)

        return Normal(mean.float(), var.sqrt().float())  # expects inputs shaped: (batch_size, d_state)

    def posterior(self, states, actions, sampling_type="random"):
        assert sampling_type in ['random', 'ensemble']
        if sampling_type == 'random':
            return self._random(states, actions)
        elif sampling_type == 'ensemble':
            return self._ensemble(states, actions)
        else:
            raise ValueError(f'Model sampling method {sampling_type} is not supported')

    def sample(self, states, actions, sampling_type="random", reparam_trick=True):
        """
        sample next states given current states and actions according to the sampling_type

        Args:
            states (torch Tensor[batch size, dim_state])
            actions (torch Tensor[batch size, dim_action])
            sampling_type (str)
            reparam_trick (bool)

        Returns:
            next state (torch Tensor[batch size, dim_state])
        """
        pdf = self.posterior(states, actions, sampling_type)
        return pdf.rsample() if reparam_trick else pdf.sample()

    def loss(self, states, actions, state_deltas):
        """
        compute loss given states, actions and state_deltas

        the loss is actually computed between predicted state delta and actual state delta, both in normalized space

        Args:
            states (torch Tensor[ensemble_size, batch size, dim_state])
            actions (torch Tensor[ensemble_size, batch size, dim_action])
            state_deltas (torch Tensor[ensemble_size, batch size, dim_state])

        Returns:
            loss (torch 0-dim Tensor, scalar): `.backward()` can be called on it to compute gradients
        """

        states, actions = self._pre_process_model_inputs(states, actions)
        targets = self._pre_process_model_targets(state_deltas)

        mu, var = self._propagate_network(states, actions)      # delta and variance

        # negative log likelihood
        loss = (mu - targets) ** 2 / var + torch.log(var)
        return torch.mean(loss)
