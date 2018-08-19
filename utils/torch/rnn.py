import abc

import torch
import torch.nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


def get_sequence_length(sequences, masking=0):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2, keepdim=False)

    masks = torch.ne(sequences, masking)
    lengths = masks.sum(dim=1, keepdim=False)

    return lengths


class BaseRNNCell(torch.nn.Module):
    def __init__(self):
        super(BaseRNNCell, self).__init__()

    @abc.abstractmethod
    def zero_state(self, batch_size):
        """Return zero state of an RNN Cell."""


def dynamic_rnn(cell, inputs, lengths):
    """Inspired by VariableRecurrent in pytorch"""
    if not isinstance(cell, BaseRNNCell):
        raise ValueError('The `cell` argument should be an instance of the BaseRNNCell class')

    inputs_packed = pack_padded_sequence(inputs, lengths, batch_first=True)
    inputs, batch_sizes = inputs_packed

    hiddens = []
    outputs = []

    input_offset = 0
    last_batch_size = batch_sizes[0]

    hidden = cell.zero_state(batch_sizes[0])

    flat_hidden = not isinstance(hidden, tuple)
    if flat_hidden:
        hidden = (hidden,)

    for batch_size in batch_sizes:
        step_input = inputs[input_offset:input_offset + batch_size]
        input_offset += batch_size

        dec = last_batch_size - batch_size
        if dec > 0:
            hiddens.append(tuple(h[-dec:] for h in hidden))
            hidden = tuple(h[:-dec] for h in hidden)
        last_batch_size = batch_size

        if flat_hidden:
            hidden = cell(step_input, hidden[0])
            hidden = (hidden,)
        else:
            hidden = cell(step_input, hidden)

        outputs.append(hidden[0])

    hiddens.append(hidden)
    hiddens.reverse()

    hiddens = tuple(torch.cat(h, 0) for h in zip(*hiddens))
    if flat_hidden:
        hiddens = hiddens[0]

    outputs = torch.cat(outputs, 0)
    outputs_packed = PackedSequence(outputs, batch_sizes)
    outputs, _ = pad_packed_sequence(outputs_packed, batch_first=True)

    return hiddens, outputs


def dynamic_rnn_reversed(cell, inputs, lengths):
    from torch.nn._functions.rnn import VariableRecurrentReverse

    if not isinstance(cell, BaseRNNCell):
        raise ValueError('The `cell` argument should be an instance of the BaseRNNCell class')

    inputs_packed = pack_padded_sequence(inputs, lengths, batch_first=True)
    inputs, batch_sizes = inputs_packed

    rnn_fn = VariableRecurrentReverse(batch_sizes, cell)
    hidden = cell.zero_state(batch_sizes[0])

    hidden, outputs = rnn_fn(inputs, hidden, [])

    outputs_packed = PackedSequence(outputs, batch_sizes)
    outputs, _ = pad_packed_sequence(outputs_packed, batch_first=True)

    return hidden, outputs
