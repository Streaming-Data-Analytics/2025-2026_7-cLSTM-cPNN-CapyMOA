import numpy as np

import torch
from torch import nn

from capymoa.base import Classifier


class cLSTMLinear(nn.Module):
    """
    Continuous LSTM with Linear output head.

    This is the core building block of cRNN and cPNN architectures.

    Args:
        input_size (int): Number of input features per timestep. Default: 2.
        device (torch.device): Device to run the model on. Default: CPU.
        hidden_size (int): Dimension of the LSTM hidden state. Default: 50.
        output_size (int): Number of output classes. Default: 2.
        batch_size (int): Batch size. Default: 128.
        many_to_one (bool): If True, only the last timestep produces output (per-instance inference mode as described in Section 1.2 of the paper).
        If False, every timestep produces output. Default: False.
    """
    def __init__(
        self,
        input_size=2,
        hidden_size=50,
        output_size=2,
        batch_size=128,
        device=torch.device('cpu'),
        many_to_one=False
    ):
        super(cLSTMLinear, self).__init__()

        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.many_to_one = many_to_one

        # Quantization
        self.quantized = False

        # Initial LSTM states stored as NumPy arrays of shape (1, hidden_size)
        self.h0 = np.zeros((1, self.hidden_size))   # Initial hidden state
        self.c0 = np.zeros((1, self.hidden_size))   # Initial cell state

        # Layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm.to(self.device)

        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.to(self.device)
        
    def forward(self, x, prev_h, train=False):
        """
        Forward pass of the cLSTM.

        Args:
            x (torch.Tensor): Input features of shape (B, L, input_size) where B = batch_size and L = sequence length
            prev_h (torch.Tensor | None): Hidden state output of shape (B, L, hidden_size)
            train (bool)

        Returns:
            out (torch.Tensor): Model predictions of shape (B, output_size) if many_to_one=True, else (B, L, output_size)
            out_h (torch.Tensor): LSTM hidden states of shape (B, L, hidden_size)
        """
        input_f = x.to(self.device)

        if prev_h is not None:
            input_f = torch.cat((x, prev_h), dim=2) # (B, L, I+H)

        out_h, _ = self.lstm(
            input_f,
            (
                self._build_initial_state(x, self.h0),
                self._build_initial_state(x, self.c0)
            )
        )

        if self.many_to_one:
            out = self.linear(out_h[:, -1:, :])[:, -1, :]
        else:
            out = self.linear(out_h)

        return out, out_h
    
    def _build_initial_state(self, x, state):
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)


class cLSTM(Classifier):
    """
    cLSTM: Continuous LSTM classifier for CapyMOA.

    Implements the cRNN training and inference lifecycle using LSTM as underlying recurrent architecture.

    Training: A tumbling window segments the data stream into mini-batches of size B. Once a mini-batch is full,
    a hopping window of size W slides over the mini-batch producing B-W+1 sequences. The model is then trained on
    such sequences for E epochs via SGD and cross-entropy loss.

    Inference: Inference is anytime and per-instance. The architecture is many-to-one, foreach incoming input vector X_t the model maintains a
    sliding window [X_{t-W+1}, ..., X_t] of length W and predicts ŷ_t from the last timestep's output.

    Args:
        schema (capymoa.stream.Schema): CapyMOA dataset schema
        window_size (int): Sequence length W (hopping window). Default: 11
        batch_size (int): Mini-batch size B (tumbling window). Default: 128
        hidden_size (int): LSTM hidden state dimension H. Default: 50
        learning_rate (float): Learning rate. Default: 0.01
        epochs (int): Training epochs E per each mini-batch. Default: 10
        random_seed (int): Random seed. Default: 1
    """

    def __init__(
        self,
        schema,
        window_size=11,
        batch_size=128,
        hidden_size=50,
        learning_rate=0.01,
        epochs=10,
        random_seed=1
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        torch.manual_seed(random_seed)

        self.W = window_size
        self.B = batch_size
        self.E = epochs

        n_features = schema.get_num_attributes()
        n_classes = schema.get_num_classes()

        # Using the cLSTMLinear module from MagicNet with many_to_one=True to use only the last timestep's output.
        self.model = cLSTMLinear(
            input_size=n_features,
            hidden_size=hidden_size,
            output_size=n_classes,
            batch_size=batch_size,
            many_to_one=True
        )

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Sliding inference window [X_{t-W+1}, ..., X_t] used for per-instance prediction
        self._inference_window = []

        # Mini-batch buffer (tumbling window) that accumulates (X_t, y_t) pairs until B pairs are available
        self._buffer_x = []
        self._buffer_y = []

    def __str__(self):
        return self.model.__str__()
    
    def _forward(self, prev_h=None):
        """
        Runs a forward pass on the current prediction window [X_{t-W+1}, ..., X_t].

        Args:
            prev_h (torch.Tensor | None): (1, W, H) lateral hidden states from previous cPNN column or None for a standalone cLSTM
        
        Returns:
            out (torch.Tensor): (1, n_classes) logits for current time step
            out_h (torch.Tensor): (1, W, H) full hidden state sequence h_{t-W+1, ..., h_t}
        """

        # If fewer than W samples have been seen so far, the left side of the window is zero-padded
        # to support anytime inference from the very first sample

        n_features = self.model.input_size
        pad_len = self.W - len(self._inference_window)

        if pad_len > 0:
            padding = [np.zeros(n_features) for _ in range(pad_len)]
            window = padding + list(self._inference_window)
        else:
            window = list(self._inference_window)

        sequence = torch.tensor(
            np.array(window),
            dtype=torch.float32
        ).unsqueeze(0)  # (1, W, n_features)

        self.model.eval()
        with torch.no_grad():
            out, out_h = self.model(x=sequence, prev_h=prev_h)

        return out, out_h

    
    def predict_proba(self, instance):
        """
        Returns the per-instance class probabilities P(y_t | X_{t-W+1},... , X_t).

        Implements the abstract method of CapyMOA's Classifier base class.
        The inherited predict() calls this method and returns argmax(probabilities).

        Slides the inference window forward by appending X_t and dropping X_{t-W}. When full, 
        then runs a many-to-one forward pass and applies softmax to the last timestep's logits.

        Args:
            instance: (capymoa.instance.Instance) CapyMOA instance carrying the feature vector X_t
        Returns:
            (numpy.ndarray) Softmax proabilities of shape (n_classes, )
        """
        # Slide the inference window by appending X_t and, if the window is full, also dropping X_{t-W}

        self._inference_window.append(instance.x.copy())
        if len(self._inference_window) > self.W:
            self._inference_window.pop(0)

        out, _ = self._forward()

        return torch.softmax(out, dim=1).squeeze(0).numpy().astype(np.float64)
    
    def train(self, instance):
        """
        Accumulates labeled pairs (X_t, y_t) into the mini-batch buffer and triggers model training
        on the current mini-bath when that mini-batch is full.

        Args:
            instance: (capymoa.instance.LabeledInstance) CapyMOA labeled instance carrying feature vector X_t and label y_t
        """
        self._buffer_x.append(instance.x.copy())
        self._buffer_y.append(instance.y_index)

        if len(self._buffer_x) >= self.B:
            self._fit()
        
            self._buffer_x = []
            self._buffer_y = []


    def _fit(self):
        """
        Trains the model on the current mini-batch for E epochs via SGD and cross-entropy loss.
        """
        n_sequences = self.B - self.W + 1

        X_batch = np.array(self._buffer_x)  # (B, n_features)
        y_batch = np.array(self._buffer_y)  # (B, )

        # Building hopping windowed sequences for the mini-batch
        X = np.array([X_batch[i : i + self.W] for i in range(n_sequences)]) # W input samples
        y = np.array([y_batch[i + self.W - 1] for i in range(n_sequences)]) # using the label of just the last sample in the sequence

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        self.model.train()
        for _ in range(self.E):
            self.optimizer.zero_grad()
            out, _ = self.model(x=X_t, prev_h=None) # no previous columns
            loss = self.criterion(out, y_t)
            loss.backward()
            self.optimizer.step()