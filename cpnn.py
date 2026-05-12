import numpy as np

import torch
from torch import nn

from capymoa.base import Classifier

from clstm import cLSTMLinear


class cPNN(Classifier):
    """
    cPNN: Continuous Progressive Neural Network classifier for CapyMOA.

    This class implements the cPNN training and inference lifecycle, extending the cLSTM
    with column expansions and lateral connections to tackle concept drifts in evolving streams
    while preserving previously acquired knowledge.

    Architecture: A list of cLSTMLinear columns, each acting as a many-to-one LSTM.
    The first column (k=0) receives the raw feature vector X_t. Every subsequent column (k>=1)
    receives the concatenation of X_t and the full hidden state sequence produced by column k-1
    at the same time step.

    Training: The last column is trained as a normal cLSTM. A tumbling window segments the data
    streams into mini-batches of size B. Once a mini-batch is full, a hopping window of size W
    produces B-W+1 sequences that are run through every column (first the frozen ones, then the last
    column) for E epochs of Adam on cross-entropy.

    Inference: Applies same many-to-one anytime inference as in a normal cLSTM. The sliding window
    [X_{t-W+1}, ..., X_t}] is propagated sequentially through all columns.

    Concept drift handling: Concept drifts on the data stream MUST trigger an external call to the
    add_new_column() API, which freezes the current active column and appends a new active one.

    Args:
        schema (capymoa.stream.Schema): CapyMOA dataset schema
        window_size (int): Sequence length W (hopping window). Default: 11
        batch_size (int): Mini-batch size B (tumbling window). Default: 128
        hidden_size (int): LSTM hidden state dimension H. Default: 50
        learning_rate (float): Learning rate. Default: 0.01
        epochs (int): Training epochs E for each mini-batch. Default: 10
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
        self.H = hidden_size
        self.lr = learning_rate

        self.n_features = schema.get_num_attributes()
        self.n_classes = schema.get_num_classes()

        # List of cPNN columns where each one is of type cLSMTLinear many_to_one.
        self.columns = nn.ModuleList()
        self.optimizers = []
        self.criterion = nn.CrossEntropyLoss()

        # Bootstrap the first column (column 0 has no lateral input)
        self._append_column(input_size=self.n_features)

        # Sliding inference window [X_{t-W+1}, ..., X_t] which is shared across columns to avoid
        # W zero-padded predictions after each concept drift.
        self._inference_window = []

        # Mini-batch buffer (tumbling window) that accumulates (X_t, y_t) pairs until B pairs are available
        # Reset by add_new_column() so the new column is trained only on post-drift data points
        self._buffer_x = []
        self._buffer_y = []

    def __str__(self):
        header = f"cPNN with {len(self.columns)} columns, W={self.W}, B={self.B}, E={self.E}, hidden={self.H}, lr={self.lr}"
        cols = '\n'.join(
            f'  Column {i}:\n    ' + col.__str__().replace('\n', '\n    ')
            for i, col in enumerate(self.columns)
        )
        return f'{header}\n{cols}'
    
    def add_new_column(self):
        """
        Expands the cPNN architecture by appending a new column after a concept drift happens.

        It freezes every parameter of the currently active last column and switches it to eval mode. A fresh
        cLSTMLinear column is then appended with input_size = n_features + H such that it can accept the
        concatenation of the raw feature vector and the previous column's full hidden state sequence. A new Adam
        optimizer is also created for the new active column. Clears the mini-batch buffer, in order to be able to
        train the new column only on post-drift data.
        """
        # Freeze every parameter of the current active column
        for param in self.columns[-1].parameters():
            param.requires_grad = False
        self.columns[-1].eval()

        # Append a new active column
        self._append_column(input_size=self.n_features + self.H)

        # Reset the mini-batch buffer
        self._buffer_x = []
        self._buffer_y = []

    def _append_column(self, input_size):
        """
        Appends a new cLSTMLinear column to self.columns and adds a fresh Adam optimizer to it

        Used both at initialization (column k=0: input_size=n_features) and at every add_new_column()
        function call (columns k>=1): input_size=n_features + H)
        """
        column = cLSTMLinear(
            input_size=input_size,
            hidden_size=self.H,
            output_size=self.n_classes,
            batch_size=self.B,
            many_to_one=True
        )

        self.columns.append(column)
        self.optimizers.append(torch.optim.Adam(params=column.parameters(), lr=self.lr))

    def _forward(self):
        """
        Runs a forward pass on the current prediction window [X_{t-W+1}, ..., X_t]
        through every column in order, propagating the lateral hidden state from each
        column into the next one.

        Returns:
            out (torch.Tensor): Logits for current time step from last column. Shape (1, n_classes)
            out_h (torch.Tensor): Hidden state sequence h_{t-w+1}, ..., h_t from last column. Shape (1, W, H)
        """
        # If less than W samples have been seen so far, the left side of the window is zero-padded
        # to support anytime inference from the very first sample

        pad_len = self.W - len(self._inference_window)

        if pad_len > 0:
            padding = [np.zeros(self.n_features) for _ in range(pad_len)]
            window = padding + list(self._inference_window)
        else:
            window = list(self._inference_window)

        sequence = torch.tensor(
            np.array(window),
            dtype=torch.float32
        ).unsqueeze(0) # (1, W, n_features)

        # During inference all columns are set to eval mode
        # The lateral hidden state from column k is fed into column k+1 as prev_h
        for column in self.columns:
            column.eval()

        with torch.no_grad():
            out, out_h = None, None
            for column in self.columns:
                out, out_h = column(x=sequence, prev_h=out_h)
        
        return out, out_h
    
    def predict_proba(self, instance):
        """
        Returns the per-instance class probabilities P(y_t | X_{t-W+1}, ..., X_t).

        Implements the abstract method of CapyMOA's Classifier base class.
        The inherited predict() method calls this method and returns argmax(probabilities).

        Slides the inference window forward by appending X_t and dropping X_{t-W} when full, then
        runs a many-to-one forward pass through every column and applies softmax to the last column's
        last-timestep logits.

        Args:
            instance (capymoa.instance.Instance) CapyMOA instance carrying the feature vector X_t
        Returns:
            (numpy.ndarray) Softmax probabilities of shape (n_classe, )
        """
        # Slide the inference window by appending X_t and, if the windows is full, also dropping X_{t-W}

        self._inference_window.append(instance.x.copy())
        if len(self._inference_window) > self.W:
            self._inference_window.pop(0)

        out, _ = self._forward()

        return torch.softmax(out, dim=1).squeeze(0).numpy().astype(np.float64)
    
    def train(self, instance):
        """
        Accumulates labeled pairs (X_t, y_t) into the mini-batch buffer and triggers training
        of the active (last) column on the current mini-batch when that mini-batch is full.

        Args:
            instance: (capymoa.instance.LabeledInstance) CapyMOA labeled instance carrying 
            feature vector X_t and label y_t
        """
        self._buffer_x.append(instance.x.copy())
        self._buffer_y.append(instance.y_index)

        if len(self._buffer_x) >= self.B:
            self._fit()

            self._buffer_x = []
            self._buffer_y = []

    def _fit(self):
        """
        Trains the active (last) column on the current mini-batch for E epochs via Adam and cross-entropy loss.

        Every column is run forward to propagate the lateral hidden state, but only the active
        column's parameters require gradients and only its optimizer is stepped. Frozen columns
        have requires_grad=False on all parameters, so backward() does not accumulate gradients
        on them regardless of whether their forward is in a no_grad context.
        """
        n_sequences = self.B - self.W + 1

        X_batch = np.array(self._buffer_x)  # (B, n_features)
        y_batch = np.array(self._buffer_y)  # (B, )

        # Building hopping windowed sequences for the mini-batch
        X = np.array([X_batch[i : i + self.W] for i in range(n_sequences)]) # W input samples
        y = np.array([y_batch[i + self.W - 1] for i in range(n_sequences)]) # using the label of just the last sample in the sequence

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        # Frozen columns stay in eval mode; only the active column is in train mode
        for column in self.columns[:-1]:
            column.eval()
        self.columns[-1].train()

        # Only the active column's optimizer is stepped. Frozen columns' parameters
        # have requires_grad=False, so backward does not accumulate gradients on them,
        # but the lateral signal still flows through the active column's weights.
        active_optimizer = self.optimizers[-1]
        for _ in range(self.E):
            active_optimizer.zero_grad()

            out, out_h = None, None
            for column in self.columns:
                out, out_h = column(x=X_t, prev_h=out_h)

            loss = self.criterion(out, y_t)
            loss.backward()
            active_optimizer.step()