import numpy as np
import torch.utils.data


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, premise, hypothesis, label, memory_key=None, memory_value=None, attention=None):
        assert len(premise) == len(hypothesis)
        self.premise = premise.astype(np.long)
        self.hypothesis = hypothesis.astype(np.long)

        if label is not None:
            assert len(hypothesis) == len(label)
            self.label = label.astype(np.long)
        else:
            self.label = None

        self.memory_key = memory_key
        self.memory_value = memory_value

        self.attention = attention

    def __getitem__(self, index):
        prem = self.premise[index]
        hyp = self.hypothesis[index]

        if self.memory_key is not None and self.memory_value is not None:
            mk = self.memory_key[index]
            mv = self.memory_value[index]
        else:
            mk = 0
            mv = 0

        if self.attention is not None:
            att = self.attention[index]
        else:
            att = 0

        # reserved for CoVe
        cove_prem = 0
        cove_hyp = 0

        if self.label is not None:
            lab = self.label[index]
        else:
            lab = 0

        return prem, hyp, mk, mv, att, cove_prem, cove_hyp, lab

    def __len__(self):
        return len(self.hypothesis)
