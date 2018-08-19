import logging

import numpy as np

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F

from models.pytorch_base_model import PyTorchBaseModel
from utils.nli import LABELS
from utils.torch.layers import Linear


class PyTorchSimpleModelModule(torch.nn.Module):
    def __init__(self, W_emb, max_len, hidden_size=300, dropout=0.2, regularization=0.000001,
                 trainable_embeddings=False, learning_rate=0.001, **kwargs):
        super(PyTorchSimpleModelModule, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.regularization = regularization
        self.trainable_embeddings = trainable_embeddings
        self.learning_rate = learning_rate

        self.W_emb = W_emb
        self.embedding_dim = W_emb.shape[1]
        self.vocab_size = W_emb.shape[0]

        # create layers
        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(W_emb))
        if not self.trainable_embeddings:
            self.embed.weight.requires_grad = False

        projection_input_dim = self.embedding_dim
        self.projection = Linear(projection_input_dim, self.hidden_size)

        self.premise_bn = torch.nn.BatchNorm1d(self.hidden_size, affine=False)
        self.hypothesis_bn = torch.nn.BatchNorm1d(self.hidden_size, affine=False)

        self.joint_dropout = torch.nn.Dropout(self.dropout)

        joint_blocks = []
        for i in range(3):
            fc = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
            nl = torch.nn.ReLU()
            dp = torch.nn.Dropout(self.dropout)
            bn = torch.nn.BatchNorm1d(2 * self.hidden_size, affine=False)
            joint_blocks.extend((fc, nl, dp, bn))

        self.joint_blocks = torch.nn.Sequential(*joint_blocks)

        self.fc_logits = torch.nn.Linear(2 * self.hidden_size, len(LABELS))

    def forward(self, premise, hypothesis, *args):
        premise = self.embed(premise)
        hypothesis = self.embed(hypothesis)

        premise = F.relu(self.projection(premise))
        hypothesis = F.relu(self.projection(hypothesis))

        premise = premise.sum(dim=1, keepdim=False)
        hypothesis = hypothesis.sum(dim=1, keepdim=False)

        premise = self.premise_bn(premise)
        hypothesis = self.hypothesis_bn(hypothesis)

        joint = torch.cat([premise, hypothesis], dim=-1)
        joint = self.joint_dropout(joint)

        joint = self.joint_blocks(joint)

        pred = self.fc_logits(joint)

        return pred


class PyTorchSimpleModel(PyTorchBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchSimpleModel, self).__init__(**kwargs)

        self.regularization = kwargs.get('regularization', 0)

        self.model = PyTorchSimpleModelModule(**kwargs)
