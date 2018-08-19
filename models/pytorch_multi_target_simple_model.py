import logging

import torch
import torch.nn
import torch.nn.functional as F

from models.pytorch_multi_target_base_model import PyTorchMultiTargetBaseModel
from utils.nli import LABELS
from utils.torch.layers import Linear


class PyTorchMultiTargetSimpleModelModule(torch.nn.Module):
    def __init__(self, W_emb, max_len, hidden_size=300, dropout=0.2, regularization=0.000001,
                 trainable_embeddings=False, learning_rate=0.001, **kwargs):
        super(PyTorchMultiTargetSimpleModelModule, self).__init__()

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

        self.projection = Linear(self.embedding_dim, self.hidden_size)

        self.premise_bn = torch.nn.BatchNorm1d(self.hidden_size, affine=False)
        self.hypothesis_bn = torch.nn.BatchNorm1d(self.hidden_size, affine=False)

        self.joint_dropout = torch.nn.Dropout(self.dropout)

        nb_shared = 1
        nb_individual = 2

        joint_blocks_shared = []
        for i in range(nb_shared):
            fc = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
            nl = torch.nn.ReLU()
            dp = torch.nn.Dropout(self.dropout)
            bn = torch.nn.BatchNorm1d(2 * self.hidden_size, affine=False)
            joint_blocks_shared.extend((fc, nl, dp, bn))
        self.joint_blocks_shared = torch.nn.Sequential(*joint_blocks_shared)

        joint_blocks_source = []
        joint_blocks_target = []
        for i in range(nb_individual):
            fc = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
            nl = torch.nn.ReLU()
            dp = torch.nn.Dropout(self.dropout)
            bn = torch.nn.BatchNorm1d(2 * self.hidden_size, affine=False)
            joint_blocks_source.extend((fc, nl, dp, bn))

            fc = torch.nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
            nl = torch.nn.ReLU()
            dp = torch.nn.Dropout(self.dropout)
            bn = torch.nn.BatchNorm1d(2 * self.hidden_size, affine=False)
            joint_blocks_target.extend((fc, nl, dp, bn))

        self.joint_blocks_source = torch.nn.Sequential(*joint_blocks_source)
        self.joint_blocks_target = torch.nn.Sequential(*joint_blocks_target)

        self.fc_logits_source = torch.nn.Linear(2 * self.hidden_size, len(LABELS))
        self.fc_logits_target = torch.nn.Linear(2 * self.hidden_size, len(LABELS))

        self.mode = None

    def forward(self, premise, hypothesis, *args):
        premise = self.embed(premise)
        hypothesis = self.embed(hypothesis)

        premise = F.relu(self.projection(premise))
        hypothesis = F.relu(self.projection(hypothesis))

        premise = premise.sum(dim=1)
        hypothesis = hypothesis.sum(dim=1)

        premise = self.premise_bn(premise)
        hypothesis = self.hypothesis_bn(hypothesis)

        joint = torch.cat([premise, hypothesis], dim=-1)
        joint = self.joint_dropout(joint)

        joint_shared = self.joint_blocks_shared(joint)

        if self.mode == 'source':
            joint_source = self.joint_blocks_source(joint_shared)
            pred_source = self.fc_logits_source(joint_source)
            return pred_source
        elif self.mode == 'target':
            joint_target = self.joint_blocks_target(joint_shared)
            pred_target = self.fc_logits_target(joint_target)
            return pred_target
        else:
            raise ValueError('You must set the `mode` for a multi-target model')


class PyTorchMultiTargetSimpleModel(PyTorchMultiTargetBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchMultiTargetSimpleModel, self).__init__(**kwargs)

        self.regularization = kwargs.get('regularization', 0)

        self.model_source = self.model_target = PyTorchMultiTargetSimpleModelModule(**kwargs)

    def train_source(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        self.model_source.mode = 'source'
        logging.info('Switching mode: %s', self.model_source.mode)
        super(PyTorchMultiTargetSimpleModel, self).train_source(data_train, data_dev, batch_size, epochs, verbose)

    def train_target(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        self.model_target.mode = 'target'
        logging.info('Switching mode: %s', self.model_target.mode)
        super(PyTorchMultiTargetSimpleModel, self).train_target(data_train, data_dev, batch_size, epochs, verbose)
