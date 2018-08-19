import logging

import torch
import torch.nn
import torch.nn.functional as F

from models.pytorch_multi_target_base_model import PyTorchMultiTargetBaseModel
from utils.nli import LABELS
from utils.torch.layers import Linear, LSTMEncoder
from utils.torch.rnn import get_sequence_length


class PyTorchMultiTargetInferSentModelModule(torch.nn.Module):
    def __init__(self, W_emb, max_len, rnn_size=300, hidden_size=300, dropout=0.2, regularization=0.000001,
                 trainable_embeddings=False, learning_rate=0.001, pool_type='max', use_umls_attention=False, **kwargs):
        super(PyTorchMultiTargetInferSentModelModule, self).__init__()

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

        self.classifier_shared = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.rnn_size * 2 * 4, self.hidden_size),
            torch.nn.Tanh(),
        )

        self.classifier_source = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size, len(LABELS)),
        )
        self.classifier_target = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(self.hidden_size, len(LABELS)),
        )

        self.mode = None

    def forward(self, premise, hypothesis, *args):
        premise_len = get_sequence_length(premise)
        hypothesis_len = get_sequence_length(hypothesis)

        premise = self.embed(premise)
        hypothesis = self.embed(hypothesis)

        premise = self.encoder(premise, premise_len)
        hypothesis = self.encoder(hypothesis, hypothesis_len)

        if self.pool_type == 'max':
            premise = torch.max(premise, dim=1, keepdim=False)[0]
            hypothesis = torch.max(hypothesis, dim=1, keepdim=False)[0]
        else:
            raise ValueError('Pool type {} is not supported'.format(self.pool_type))

        features = torch.cat([premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis], dim=-1)

        features_shared = self.classifier_shared(features)

        if self.mode == 'source':
            pred = self.classifier_source(features_shared)
        elif self.mode == 'target':
            pred = self.classifier_target(features_shared)
        else:
            raise ValueError('You must set the `mode` for a multi-target model')

        return pred


class PyTorchMultiTargetInferSentModel(PyTorchMultiTargetBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchMultiTargetInferSentModel, self).__init__(**kwargs)

        self.regularization = kwargs.get('regularization', 0)

        self.model_source = self.model_target = PyTorchMultiTargetInferSentModelModule(**kwargs)

    def train_source(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        self.model_source.mode = 'source'
        logging.info('Switching mode: %s', self.model_source.mode)
        super(PyTorchMultiTargetInferSentModel, self).train_source(data_train, data_dev, batch_size, epochs, verbose)

    def train_target(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        self.model_target.mode = 'target'
        logging.info('Switching mode: %s', self.model_target.mode)
        super(PyTorchMultiTargetInferSentModel, self).train_target(data_train, data_dev, batch_size, epochs, verbose)
