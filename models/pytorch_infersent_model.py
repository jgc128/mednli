import logging

import numpy as np

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch_base_model import PyTorchBaseModel
from utils.nli import LABELS
from utils.torch.layers import Linear, LSTMEncoder, InterSentenceInteraction
from utils.torch.rnn import get_sequence_length


class PyTorchInferSentModelModule(torch.nn.Module):
    def __init__(self, W_emb, max_len, rnn_size=300, hidden_size=300, dropout=0.2, regularization=0.000001,
                 trainable_embeddings=False, learning_rate=0.001, pool_type='max', use_umls_attention=False, **kwargs):
        super(PyTorchInferSentModelModule, self).__init__()

        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
        self.pool_type = pool_type
        self.dropout = dropout
        self.regularization = regularization
        self.trainable_embeddings = trainable_embeddings
        self.learning_rate = learning_rate

        self.use_umls_attention = use_umls_attention

        self.W_emb = W_emb
        self.embedding_dim = W_emb.shape[1]
        self.vocab_size = W_emb.shape[0]

        # create layers
        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(W_emb))
        if not self.trainable_embeddings:
            self.embed.weight.requires_grad = False

        self.encoder = LSTMEncoder(self.embedding_dim, self.rnn_size, bidirectional=True, return_sequence=True)

        if self.use_umls_attention:
            self.inter_sentence_interaction = InterSentenceInteraction()
        #     self.attention_projection = torch.nn.Linear(self.rnn_size * 2, self.rnn_size * 2)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.rnn_size * 2 * 4, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size, len(LABELS)),
        )

    def forward(self, premise, hypothesis, *args):
        premise_len = get_sequence_length(premise)
        hypothesis_len = get_sequence_length(hypothesis)

        premise = self.embed(premise)
        hypothesis = self.embed(hypothesis)

        premise = self.encoder(premise, premise_len)
        hypothesis = self.encoder(hypothesis, hypothesis_len)

        if self.use_umls_attention:
            attention = args[2]
            attention = attention[:, :premise.size(1), :hypothesis.size(1)]
            premise_att, hypothesis_att = self.inter_sentence_interaction(premise, hypothesis, attention)

            # premise_att = F.elu(self.attention_projection(premise_att))
            # hypothesis_att = F.elu(self.attention_projection(hypothesis_att))

            premise = torch.cat([premise, premise_att], dim=1)
            hypothesis = torch.cat([hypothesis, hypothesis_att], dim=1)

        if self.pool_type == 'max':
            premise = torch.max(premise, dim=1, keepdim=False)[0]
            hypothesis = torch.max(hypothesis, dim=1, keepdim=False)[0]
        else:
            raise ValueError('Pool type {} is not supported'.format(self.pool_type))

        features = torch.cat([premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis], dim=-1)

        pred = self.classifier(features)

        return pred


class PyTorchInferSentModel(PyTorchBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchInferSentModel, self).__init__(**kwargs)

        self.regularization = kwargs.get('regularization', 0)

        self.model = PyTorchInferSentModelModule(**kwargs)
