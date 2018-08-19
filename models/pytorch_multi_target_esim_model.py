import torch
import torch.nn
import torch.nn.functional as F

from models.pytorch_esim_model import PyTorchESIMModule
from models.pytorch_multi_target_base_model import PyTorchMultiTargetBaseModel
from utils.nli import LABELS


class PyTorchMultiTargetESIMModelModule(torch.nn.Module):
    def __init__(self, esim):
        super(PyTorchMultiTargetESIMModelModule, self).__init__()

        self.esim = esim

        self.fc_representation = torch.nn.Linear(self.esim.hidden_size * 4 * 2, self.esim.hidden_size)
        self.fc_logits = torch.nn.Linear(self.esim.hidden_size, len(LABELS))

    def forward(self, premise, hypothesis, *args):
        v = self.esim(premise, hypothesis)

        v = F.tanh(self.fc_representation(v))

        pred = self.fc_logits(v)

        return pred


class PyTorchMultiTargetESIMModel(PyTorchMultiTargetBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchMultiTargetESIMModel, self).__init__(**kwargs)

        self.regularization = kwargs.get('regularization', 0)

        self.esim = PyTorchESIMModule(**kwargs)

        self.model_source = PyTorchMultiTargetESIMModelModule(self.esim)
        self.model_target = PyTorchMultiTargetESIMModelModule(self.esim)
