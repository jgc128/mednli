import logging

import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.torch.rnn import BaseRNNCell, dynamic_rnn, dynamic_rnn_reversed


class Bottle(torch.nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, torch.nn.Linear):
    pass


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, return_sequence=False):
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequence = return_sequence
        self.bidirectional = bidirectional

        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                                 batch_first=True)

    def zero_state(self, inputs):
        batch_size = inputs.size()[0]

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 1 if not self.bidirectional else 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        # will work on both GPU and CPU in contrast to just Variable(*state_shape)
        h0, c0 = (
            Variable(inputs.data.new(*state_shape).zero_()),
            Variable(inputs.data.new(*state_shape).zero_())
        )

        return h0, c0

    def forward(self, inputs, lengths=None):
        h0, c0 = self.zero_state(inputs)

        if lengths is not None:
            # sort by length
            lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
            inputs_sorted = inputs[inputs_sorted_idx]

            # pack sequences
            packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, list(lengths_sorted.data), batch_first=True)

            outputs, (h, c) = self.rnn(packed, (h0, c0))

            # unpack sequences
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            # un-sort
            _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
            outputs = outputs[inputs_unsorted_idx]

            # concat in case of bidirectional, and just remove the first dim in case of unidirectional
            h = torch.cat(h, dim=-1)
            h = h[inputs_unsorted_idx]
        else:
            outputs, (h, c) = self.rnn(inputs, (h0, c0))

            # concat in case of bidirectional, and just remove the fisrt dim in case of unidirectional
            h = torch.cat(h, dim=-1)

        if self.return_sequence:
            return outputs
        else:
            return h


class InterSentenceAttention(torch.nn.Module):
    def forward(self, a, b):
        assert a.size()[2] == b.size()[2]
        assert a.size()[0] == b.size()[0]

        # performs batch mat mul
        attention = torch.matmul(
            a,
            b.transpose(1, 2),
        )

        return attention


def softmax(input, axis=1):
    """https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637"""
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)


class InterSentenceInteraction(torch.nn.Module):
    def __init__(self):
        super(InterSentenceInteraction, self).__init__()

        self._attention = InterSentenceAttention()

    def forward(self, a, b, e=None):
        if e is None:
            e = self._attention(a, b)

        e_b = softmax(e, axis=2)
        e_a = softmax(e, axis=1)

        a_tilda = torch.matmul(e_b, b)
        b_tilda = torch.matmul(e_a.transpose(2, 1), a)

        return a_tilda, b_tilda


class KeyValueMemoryLookup(torch.nn.Module):
    def forward(self, sentence, memory_key, memory_value):
        squeeze = False
        if len(sentence.size()) == 2:
            sentence = sentence.unsqueeze(1)
            squeeze = True

        sentence_normalized = F.normalize(sentence, p=2, dim=-1)
        memory_key_normalized = F.normalize(memory_key, p=2, dim=-1)

        mem_attention = torch.matmul(sentence_normalized,
                                     memory_key_normalized.transpose(1, 2))  # (batch_size, max_len, mem_size)
        mem_attention = softmax(mem_attention, axis=2)

        mem_attended = mem_attention.unsqueeze(3) * memory_value.unsqueeze(1)  # (?, max_len, mem_size, emd_dim)
        mem_attended = mem_attended.sum(dim=2)  # (?, max_len, emd_dim)

        if squeeze:
            mem_attended = mem_attended.squeeze()

        return mem_attended


class InteractionEnhancement(torch.nn.Module):
    def __init__(self, extended=True):
        super(InteractionEnhancement, self).__init__()

        self.extended = extended

    def forward(self, *args):
        to_concat = []
        to_concat.extend(args)

        if self.extended:
            a0 = args[0]
            for a1 in args[1:]:
                to_concat.append(a0 - a1)
                to_concat.append(a0 * a1)

        m_a = torch.cat(to_concat, dim=-1)

        return m_a


class MaxAvgPool(torch.nn.Module):
    def forward(self, inputs, lengths=None):
        inputs_max, _ = inputs.max(dim=1, keepdim=False)

        if lengths is None:
            inputs_avg = inputs.mean(dim=1, keepdim=False)
        else:
            inputs_avg = inputs.sum(dim=1, keepdim=False) / lengths.float().unsqueeze(1)

        result = torch.cat([inputs_avg, inputs_max], dim=-1)

        return result


class DynamicMemoryCell(BaseRNNCell):
    def __init__(self, nb_slots=4, memory_size=300):
        super(DynamicMemoryCell, self).__init__()

        self.nb_slots = nb_slots
        self.memory_size = memory_size

        self.keys = torch.nn.ParameterList()
        for i in range(nb_slots):
            key = torch.nn.Parameter(torch.randn(memory_size), requires_grad=False)
            key.data.uniform_(-0.1, 0.1)
            self.keys.append(key)

        self.U = torch.nn.Parameter(torch.randn(memory_size, memory_size))
        self.V = torch.nn.Parameter(torch.randn(memory_size, memory_size))
        self.W = torch.nn.Parameter(torch.randn(memory_size, memory_size))
        torch.nn.init.xavier_normal(self.U)
        torch.nn.init.xavier_normal(self.V)
        torch.nn.init.xavier_normal(self.W)

        self.gate_bias = torch.nn.Parameter(torch.zeros(1))
        self.state_bias = torch.nn.Parameter(torch.zeros(1, memory_size))

        logging.info('Creating memory: [%s, %s]', self.nb_slots, self.memory_size)

    def zero_state(self, batch_size):
        states = tuple([
            Variable(self.keys[i].data).unsqueeze(0).expand(batch_size, self.memory_size)
            for i in range(self.nb_slots)
        ])
        return states

    def forward(self, inputs, states):
        # inputs have the shape of (?, mem_size)
        # states is a list which elements have the shape of (?, mem_size)

        new_states = []
        for j, state in enumerate(states):
            key = self.keys[j]

            g_state = torch.matmul(inputs.unsqueeze(1), state.unsqueeze(2)).squeeze()  # shape (?,)
            g_key = torch.matmul(inputs, key)  # shape (?,)
            g = F.sigmoid(g_state + g_key + self.gate_bias)  # shape (?,)

            # get h_tilda
            h_tilda_state = torch.matmul(state, self.U)  # shape (?, mem_size)
            h_tilda_key = torch.matmul(key.unsqueeze(0), self.V)  # shape (1, mem_size)
            h_tilda_inputs = torch.matmul(inputs, self.W)  # shape (?, mem_size)
            h_tilda = F.elu(h_tilda_state + h_tilda_key + h_tilda_inputs + self.state_bias)  # shape (?, mem_size)

            h = state + g.unsqueeze(1) * h_tilda  # shape (?, mem_size)

            # normalize
            h = F.normalize(h, p=2, dim=-1)

            new_states.append(h)

        new_states = tuple(new_states)
        return new_states


class DynamicMemory(torch.nn.Module):
    """Based on https://github.com/siddk/entity-network"""

    def __init__(self, nb_slots=4, memory_size=300, bidirectional=False, **kwargs):
        super(DynamicMemory, self).__init__()

        self.cell = DynamicMemoryCell(nb_slots, memory_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, lengths):
        lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[inputs_sorted_idx]
        inputs_lenghts = list(lengths_sorted.data)

        hiddens, outputs = dynamic_rnn(self.cell, inputs_sorted, inputs_lenghts)
        hiddens = torch.stack(hiddens, dim=1)

        if self.bidirectional:
            hiddens_rev, outputs_rev = dynamic_rnn_reversed(self.cell, inputs_sorted, inputs_lenghts)
            hiddens_rev = torch.stack(hiddens_rev, dim=1)

            hiddens = torch.cat([hiddens, hiddens_rev], dim=-1)

        _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
        hiddens = hiddens[inputs_unsorted_idx]

        return hiddens


Variable.__repr__ = lambda x: 'Variable {}'.format(tuple(x.size()))
torch.nn.Parameter.__repr__ = lambda x: 'Parameter {}'.format(tuple(x.size()))
