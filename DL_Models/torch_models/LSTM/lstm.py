import torch
import torch.nn as nn
from config import config
from DL_Models.torch_models.LstmNetTorch import LSTMNet


class LSTMModel2(LSTMNet):
    def __init__(
        self,
        input_shape,
        output_shape,
        loss,
        model_number,
        batch_size,
        epochs,
        verbose,
        n_layers=1,
        hidden_dim=64,
        drop_prob=0.6,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.output_shape = output_shape
        self.nb_features = 16  # arbitrary right now
        super().__init__(
            loss=loss,
            model_number=model_number,
            batch_size=batch_size,
            input_shape=input_shape,
            output_size=output_shape,
            epochs=epochs,
            verbose=verbose,
            use_residual=False,
            hidden_dim=hidden_dim,
            drop_prob=drop_prob,
            n_layers=n_layers,
        )

    def _module(self):
        return nn.Sequential(
            nn.LSTM(
                input_size=self.input_shape[0],
                hidden_size=self.hidden_dim,
                num_layers=self.n_layers,
                dropout=self.drop_prob,
                batch_first=True,
            ),
            nn.Dropout(p=self.drop_prob),
            nn.Linear(
                in_features=self.hidden_dim, out_features=self.output_shape
            ),
            nn.Sigmoid(),
        )
