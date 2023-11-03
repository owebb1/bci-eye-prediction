from abc import ABC, abstractmethod
from re import X

from DL_Models.torch_models.BaseNetTorch import BaseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from torch.autograd import Variable
import logging
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool


class LSTMNet(ABC, BaseNet):
    """
    This class defines all the common functionality for convolutional nets
    Inherit from this class and only implement _module() and
    _get_nb_features_output_layer() methods
    Modules are then stacked in the forward() pass of the model
    """

    def __init__(
        self,
        loss,
        model_number,
        batch_size,
        input_shape,
        output_size,
        epochs,
        verbose,
        use_residual,
        hidden_dim,
        drop_prob,
        n_layers,
    ):

        # super().__init__(
        #     loss=loss,
        #     input_shape=input_shape,
        #     output_shape=output_size,
        #     epochs=epochs,
        #     verbose=verbose,
        #     model_number=model_number,
        # )
        # self.hidden_layers = hidden_dim
        # self.input_size = input_shape[0]
        # lstm1, lstm2, linear are all layers in the network
        # self.lstm = nn.LSTM(
        #     self.input_size, self.hidden_layers, 1, batch_first=True
        # )
        # self.linear = nn.Linear(hidden_dim, output_size)

        super().__init__(
            loss=loss,
            input_shape=input_shape,
            output_shape=output_size,
            epochs=epochs,
            verbose=verbose,
            model_number=model_number,
        )

        self.num_classes = 2  # number of classes
        self.num_layers = 1  # number of layers
        self.input_size = input_shape[0]  # input size
        # self.nb_channels = input_shape[1]
        self.hidden_size = hidden_dim  # hidden state
        self.batch_size = batch_size
        self.hidden = None

        self.lstm = nn.LSTM(
            self.input_size,  # 500
            self.hidden_size,  # 64
            self.num_layers,  # 1
            batch_first=True,
        )
        # self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.fc_1 = nn.Linear(self.hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(
            128, self.num_classes
        )  # fully connected last layer

        self.relu = nn.ReLU()
        self.swish = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        if self.verbose and model_number == 0:
            logging.info(
                f"Number of model parameters: {sum(p.numel() for p in self.parameters())}"
            )
            logging.info(
                f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
            )
            logging.info(
                "--------------- input size : " + str(self.input_size)
            )
            logging.info(
                "--------------- hidden size : " + str(self.hidden_size)
            )
            logging.info(
                "--------------- nb channels : " + str(self.nb_channels)
            )
            logging.info(
                "--------------- output size : " + str(self.num_classes)
            )
            logging.info(
                "--------------- number of layers : " + str(self.num_layers)
            )
            logging.info(
                "--------------- batch_size        : " + str(self.batch_size)
            )
            logging.info("--------------- preprocessing : " + str(False))

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.nb_channels, self.hidden_size),
            torch.zeros(self.num_layers, self.nb_channels, self.hidden_size),
        )

    def forward(self, x):
        """
        Implements the forward pass of the network
        """

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device=x.device)

        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,device=x.device)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        out = hn.view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next
        out = self.swish(out)
        out = self.fc_1(out)  # Final Output
        out = self.swish(out)
        out = self.fc(out)
        #out = self.sigmoid(out)
        return out

        # h_0 = Variable(
        #     torch.zeros(self.num_layers, 64, self.hidden_size)
        # )  # hidden state
        # c_0 = Variable(
        #     torch.zeros(self.num_layers, 64, self.hidden_size)
        # )  # internal state
        # # Propagate input through LSTM
        # output, (hn, cn) = self.lstm(
        #     x, (h_0, c_0)
        # )  # lstm with input, hidden, and internal state
        # output = hn.view(
        #     -1, self.hidden_size
        # )  # reshaping the data for Dense layer next
        # out = self.relu(output)
        # out = self.fc_1(out)  # first Dense
        # out = self.relu(out)  # relu
        # out = self.fc(out)  # Final Output

        # return out
        # out = output[:, -1]
        # h0 = torch.zeros(
        #     self.num_layers, self.nb_channels, self.hidden_size
        # ).requires_grad_()

        # # Initialize cell state
        # c0 = torch.zeros(
        #     self.num_layers, self.nb_channels, self.hidden_size
        # ).requires_grad_()

        # # 28 time steps
        # # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # # If we don't, we'll backprop all the way to the start even after going through another batch
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # # Index hidden state of last time step
        # # out.size() --> 100, 28, 100
        # # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        # out = self.fc(out[:, -1, :])
        # # out.size() --> 100, 10
        # return out

        out, self.hidden = self.lstm(x, self.hidden)

        # Index hidden state of last time step
        # out.size() --> 64, 129, 64
        # print(out.size())

        # out[:, -1, :] --> 64, 64 --> just want last time step hidden states!
        out_fc = self.fc(out)
        out_vals = out_fc[:, -1, :]
        # out.size() --> 64, 2
        return out_vals

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network
        """
        return self.nb_features * self.timesamples

    # abstract method
    def _preprocessing(self, input_tensor):
        pass

    # abstract method
    def _module(self, input_tensor, current_depth):
        pass

    def _split_model(self):
        pass
