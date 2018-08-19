import logging

import numpy as np

import torch
import torch.nn
import torch.utils.data
import torch.functional
import torch.nn.functional as F
from torch.autograd import Variable

from models.pytorch_base_model import PyTorchBaseModel
from utils.nli import LABELS
from utils.torch.layers import Linear, LSTMEncoder, InterSentenceAttention, softmax, InterSentenceInteraction, \
    InteractionEnhancement, MaxAvgPool
from utils.torch.rnn import get_sequence_length


class PyTorchESIMModule(torch.nn.Module):
    def __init__(self, W_emb, rnn_size=300, hidden_size=300, dropout=0.2, regularization=0.000001,
                 trainable_embeddings=False, learning_rate=0.001, use_umls_attention=False, **kwargs):
        super(PyTorchESIMModule, self).__init__()

        self.rnn_size = rnn_size
        self.hidden_size = hidden_size
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

        input_encoder_dim = self.embedding_dim

        self.input_encoder = LSTMEncoder(input_encoder_dim, self.rnn_size, return_sequence=True, bidirectional=True)

        self.inter_sentence_interaction = InterSentenceInteraction()
        self.interaction_enhancement = InteractionEnhancement()

        self.interaction_enhancement_mapping = Linear(self.rnn_size * 4 * 2, self.hidden_size)

        rnn_composition_dim = self.hidden_size if not self.use_umls_attention else self.hidden_size * 2
        self.rnn_composition = LSTMEncoder(rnn_composition_dim, self.rnn_size, return_sequence=True, bidirectional=True)

        self.max_avg_pool = MaxAvgPool()

    def forward(self, premise, hypothesis, *args):
        premise_len = get_sequence_length(premise)
        hypothesis_len = get_sequence_length(hypothesis)

        # =============================
        # === input encoding ==========
        premise = self.embed(premise)
        hypothesis = self.embed(hypothesis)

        a_bar = self.input_encoder(premise, premise_len)
        b_bar = self.input_encoder(hypothesis, hypothesis_len)

        # =============================
        # === local inference =========
        a_tilda, b_tilda = self.inter_sentence_interaction(a_bar, b_bar)

        if self.use_umls_attention:
            attention = args[2]
            attention = attention[:, :a_bar.size(1), :b_bar.size(1)]
            a_tilda_att, b_tilda_att = self.inter_sentence_interaction(a_bar, b_bar, attention)

        # enhancement
        m_a = self.interaction_enhancement(a_bar, a_tilda)
        m_b = self.interaction_enhancement(b_bar, b_tilda)

        if self.use_umls_attention:
            m_a_att = self.interaction_enhancement(a_bar, a_tilda_att)
            m_b_att = self.interaction_enhancement(b_bar, b_tilda_att)
        # =============================
        # === inference composition ===
        m_a_f = F.relu(self.interaction_enhancement_mapping(m_a))
        m_b_f = F.relu(self.interaction_enhancement_mapping(m_b))

        if self.use_umls_attention:
            m_a_f_att = F.relu(self.interaction_enhancement_mapping(m_a_att))
            m_b_f_att = F.relu(self.interaction_enhancement_mapping(m_b_att))

            m_a_f = torch.cat([m_a_f, m_a_f_att], dim=-1)
            m_b_f = torch.cat([m_b_f, m_b_f_att], dim=-1)

        # composition with an RNN
        v_a = self.rnn_composition(m_a_f, premise_len)
        v_b = self.rnn_composition(m_b_f, hypothesis_len)

        # pooling
        v_a = self.max_avg_pool(v_a, premise_len)
        v_b = self.max_avg_pool(v_b, hypothesis_len)
        v = torch.cat([v_a, v_b], dim=-1)

        return v


class PyTorchESIMModelModule(torch.nn.Module):
    def __init__(self, W_emb, rnn_size=300, hidden_size=300, dropout=0.2, regularization=0.000001,
                 trainable_embeddings=False, learning_rate=0.001, **kwargs):
        super(PyTorchESIMModelModule, self).__init__()

        self.esim = PyTorchESIMModule(W_emb, rnn_size, hidden_size, dropout, regularization,
                                      trainable_embeddings, learning_rate, **kwargs)

        self.fc_representation = torch.nn.Linear(self.esim.rnn_size * 4 * 2, self.esim.hidden_size)
        self.fc_logits = torch.nn.Linear(self.esim.hidden_size, len(LABELS))

    def forward(self, premise, hypothesis, *args):
        v = self.esim(premise, hypothesis, *args)

        v = F.tanh(self.fc_representation(v))

        pred = self.fc_logits(v)

        return pred


class PyTorchESIMModel(PyTorchBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchESIMModel, self).__init__(**kwargs)

        self.regularization = kwargs.get('regularization', 0)

        self.model = PyTorchESIMModelModule(**kwargs)
