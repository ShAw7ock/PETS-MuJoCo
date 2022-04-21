import numpy as np
import torch


class Buffer:
    def __init__(self, d_state, d_action, size):
        """
        data buffer that holds transitions

        Args:
            d_state: dimensionality of state
            d_action: dimensionality of action
            size: maximum number of transitions to be stored (memory allocated at init)
        """
        # Dimensions
        self.size = size
        self.d_state = d_state
        self.d_action = d_action

        # Main Attributes
        self.states = torch.zeros(size, d_state).float()
        self.actions = torch.zeros(size, d_action).float()
        self.state_deltas = torch.zeros(size, d_state).float()
        self.rewards = torch.zeros(size, 1).float()

        # Other attributes
        self.normalizer = None
        self.ptr = 0
        self.is_full = False

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _add(self, buffer, arr):
        n = arr.size(0)
        excess = self.ptr + n - self.size  # by how many elements we exceed the size
        if excess <= 0:  # all elements fit
            a, b = n, 0
        else:
            a, b = n - excess, excess  # we need to split into a + b = n; a at the end and the rest in the beginning
        buffer[self.ptr:self.ptr + a] = arr[:a]
        buffer[:b] = arr[a:]

    def add(self, states, actions, next_states, rewards):
        """
        add transition(s) to the buffer

        Args:
            states: pytorch Tensors of (n_transitions, d_state) shape
            actions: pytorch Tensors of (n_transitions, d_action) shape
            next_states: pytorch Tensors of (n_transitions, d_state) shape
        """
        states, actions, next_states, rewards = [x.clone().cpu() for x in [states, actions, next_states, rewards]]

        state_deltas = next_states - states
        n_transitions = states.size(0)

        assert n_transitions <= self.size

        self._add(self.states, states)
        self._add(self.actions, actions)
        self._add(self.state_deltas, state_deltas)
        self._add(self.rewards, rewards)

        if self.ptr + n_transitions >= self.size:
            self.is_full = True

        self.ptr = (self.ptr + n_transitions) % self.size

        if self.normalizer is not None:
            for s, a, ns, r in zip(states, actions, state_deltas, rewards):
                self.normalizer.add(s, a, ns, r)

    def view(self):
        n = len(self)

        s = self.states[:n]
        a = self.actions[:n]
        s_delta = self.state_deltas[:n]
        ns = s + s_delta

        return s, a, ns, s_delta

    def train_batches(self, ensemble_size, batch_size):
        """
        return an iterator of batches

        Args:
            batch_size: number of samples to be returned
            ensemble_size: size of the ensemble

        Returns:
            state of size (ensemble_size, n_samples, d_state)
            action of size (ensemble_size, n_samples, d_action)
            next state of size (ensemble_size, n_samples, d_state)
        """
        num = len(self)
        indices = [np.random.permutation(range(num)) for _ in range(ensemble_size)]
        indices = np.stack(indices)

        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop the last incomplete batch
                return

            batch_size = j - i

            batch_indices = indices[:, i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = states.reshape(ensemble_size, batch_size, self.d_state)
            actions = actions.reshape(ensemble_size, batch_size, self.d_action)
            state_deltas = state_deltas.reshape(ensemble_size, batch_size, self.d_state)

            yield states, actions, state_deltas

    def sample(self, batch_size, device):
        """
        This function will only sample the data with size batch_size.

        Args:
            batch_size: number of samples to be returned
            device: torch.Device

        Returns:
            state of size (n_samples, d_state)
            action of size (n_samples, d_action)
            next state of size (n_samples, d_state)
            reward of size (n_samples, 1)
        """
        curr_size = len(self)
        sample_size = min(curr_size, batch_size)
        indices = np.random.randint(0, curr_size, sample_size)

        states = self.states[indices].reshape(sample_size, self.d_state).to(device)
        actions = self.actions[indices].reshape(sample_size, self.d_action).to(device)
        state_deltas = self.state_deltas[indices].reshape(sample_size, self.d_state).to(device)
        rewards = self.rewards[indices].reshape(sample_size, 1).to(device)

        return states, actions, states + state_deltas, rewards

    def __len__(self):
        return self.size if self.is_full else self.ptr

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        # backward compatibility with old buffers
        if 'size' not in state and 'ptr' not in state and 'is_full' not in state:
            self.size = state['buffer_size']
            self.ptr = state['_n_elements'] % state['buffer_size']
            self.is_full = (state['_n_elements'] > state['buffer_size'])
            del self.buffer_size
            del self._n_elements
            del self.ensemble_size


class SequenceBuffer:
    def __init__(self, d_state, d_action, max_episode_steps, device):
        self.device = device
        # Main storage
        self.states = torch.zeros(max_episode_steps, d_state).to(device)
        self.actions = torch.zeros(max_episode_steps, d_action).to(device)
        self.rewards = torch.zeros(max_episode_steps, 1).to(device)

        self.max_size = max_episode_steps
        self.ptr = 0

    def __len__(self):
        return self.ptr

    def add(self, state, action, reward):
        """
        Args:
            state: pytorch Tensors of (n_transitions, d_state) shape
            action: pytorch Tensors of (n_transitions, d_action) shape
            reward: pytorch Tensors of (n_transitions, d_state) shape
        """
        state, action, reward = [x.clone().to(self.device) for x in [state, action, reward]]
        n = state.size(0)
        excess = self.ptr + n - self.max_size
        # excess > 0 sequence overflow, give up the incoming data
        if excess <= 0:
            self.states[self.ptr: self.ptr + n] = state
            self.actions[self.ptr: self.ptr + n] = action
            self.rewards[self.ptr: self.ptr + n] = reward

            self.ptr += n

    def sample(self, sample_size=None):
        """
        Sample the states in sequence for model predicting.
        Returns:
            states: [sample_size, d_state]
        """
        sample_size = self.ptr if sample_size is None else sample_size
        states = self.states[: sample_size]
        actions = self.actions[: sample_size]
        rewards = self.rewards[: sample_size]
        return states, actions, rewards

    def clear(self):
        self.ptr = 0
