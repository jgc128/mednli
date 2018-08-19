import logging
from functools import lru_cache

from tqdm import tqdm

import numpy as np

import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from models.base_model import BaseModel
from utils.torch import NLIDataset


class PyTorchBaseModel(BaseModel):
    def __init__(self, checkpoint_filename=None, training_loop_mode='best_acc', **kwargs):
        super(PyTorchBaseModel, self).__init__()

        if training_loop_mode not in {'best_acc', 'best_loss'}:
            raise ValueError('Training loop mode {} is unknown'.format(training_loop_mode))

        self.checkpoint_filename = checkpoint_filename
        self.training_loop_mode = training_loop_mode

        self.regularization = 0

        logging.info('Training loop mode: %s', self.training_loop_mode)

    def build(self):
        pass

    def _get_trainable_parameters(self):
        parameters = list(self.model.parameters())
        nb_params_before = sum(p.nelement() for p in parameters)

        parameters = [p for p in parameters if p.requires_grad]
        nb_params_after = sum(p.nelement() for p in parameters)

        logging.info('Parameters: %s -> %s', nb_params_before, nb_params_after)
        return parameters

    @property
    @lru_cache()
    def __use_gpu(self):
        gpu_device_ids = list(range(torch.cuda.device_count()))
        logging.info('Available GPUs: %s', gpu_device_ids)
        use_gpu = len(gpu_device_ids) > 0
        return use_gpu

    def train(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        # determine if we need to use gpu

        data_set_train = NLIDataset(**data_train)
        data_loader_train = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True,
                                                        num_workers=2, pin_memory=self.__use_gpu)

        if data_dev is not None:
            data_set_dev = NLIDataset(**data_dev)
            data_loader_dev = torch.utils.data.DataLoader(data_set_dev, batch_size=batch_size, shuffle=False,
                                                          num_workers=2, pin_memory=self.__use_gpu)
        else:
            data_loader_dev = []

        data_loaders = {
            'train': data_loader_train,
            'dev': data_loader_dev,
        }
        data_size = {
            'train': len(data_train['premise']),
            'dev': len(data_dev['premise']),
        }

        loss_fn = torch.nn.CrossEntropyLoss()

        parameters = self._get_trainable_parameters()
        optimizer = torch.optim.Adam(parameters, weight_decay=self.regularization)

        if self.__use_gpu:
            self.model.cuda()
            loss_fn.cuda()

        j = 1
        val_best = 0 if self.training_loop_mode == 'best_acc' else np.inf
        early_stopping_patience = 5
        improvement_was_back_to = 0
        for epoch in range(epochs):
            for phase in ['train', 'dev']:
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                running_loss = 0.0
                running_corrects = 0.0

                if verbose > 1:
                    progress_bar = tqdm(desc='{}'.format(phase), total=len(data_loaders[phase].dataset), ncols=120)
                else:
                    progress_bar = None

                for j, args in enumerate(data_loaders[phase], 1):
                    inputs = args[:-1]
                    label = args[-1]
                    if self.__use_gpu:
                        inputs = [inpt.cuda() for inpt in inputs]
                        label = label.cuda()

                    inputs = [Variable(inpt) for inpt in inputs]
                    label = Variable(label)

                    optimizer.zero_grad()

                    outputs = self.model(*inputs)
                    _, predictions = torch.max(outputs.data, dim=1, keepdim=False)

                    loss = loss_fn(outputs, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.data[0]
                    running_corrects += torch.sum(predictions == label.data)

                    if progress_bar is not None:
                        progress_bar.update(len(inputs[0]))

                epoch_loss = running_loss / j
                epoch_acc = running_corrects / data_size[phase]

                if progress_bar is not None:
                    progress_bar.close()

                log_str = 'Epoch {:<3} {:>7} loss {:.5f} accuracy {:.3f}'.format(epoch, phase, epoch_loss, epoch_acc)

                if phase == 'dev':
                    is_new_best_acc = epoch_acc > val_best
                    is_new_best_loss = epoch_loss < val_best
                    need_save_by_acc = self.training_loop_mode == 'best_acc' and is_new_best_acc
                    need_save_by_loss = self.training_loop_mode == 'best_loss' and is_new_best_loss

                    if need_save_by_acc or need_save_by_loss:
                        torch.save(self.model.state_dict(), str(self.checkpoint_filename))
                        val_best = epoch_acc if self.training_loop_mode == 'best_acc' else epoch_loss
                        improvement_was_back_to = 0

                        log_str += ' [model saved]'
                    else:
                        improvement_was_back_to += 1

                logging.info(log_str)

                if phase == 'dev' and improvement_was_back_to >= early_stopping_patience:
                    return

    def evaluate(self, data, batch_size=512):
        data_set = NLIDataset(**data)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=2, pin_memory=self.__use_gpu)

        loss_fn = torch.nn.CrossEntropyLoss()

        if self.__use_gpu:
            self.model.cuda()

        self.model.train(False)

        j = 1
        running_loss = 0.0
        running_corrects = 0.0
        for j, args in enumerate(data_loader, 1):
            inputs = args[:-1]
            label = args[-1]
            if self.__use_gpu:
                inputs = [inpt.cuda() for inpt in inputs]
                label = label.cuda()

            inputs = [Variable(inpt) for inpt in inputs]
            label = Variable(label)

            outputs = self.model(*inputs)
            _, predictions = torch.max(outputs.data, 1)

            loss = loss_fn(outputs, label)

            running_loss += loss.data[0]
            running_corrects += torch.sum(predictions == label.data)

        epoch_loss = running_loss / j
        epoch_acc = running_corrects / len(data_set)

        return epoch_loss, epoch_acc

    def predict(self, data, batch_size=512, verbose=1):
        data_set = NLIDataset(**data)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=2, pin_memory=self.__use_gpu)

        if self.__use_gpu:
            self.model.cuda()

        self.model.train(False)

        predictions = []
        for j, args in enumerate(data_loader, 1):
            inputs = args[:-1]
            if self.__use_gpu:
                inputs = [inpt.cuda() for inpt in inputs]

            inputs = [Variable(inpt) for inpt in inputs]

            outputs = self.model(*inputs)

            predictions.append(outputs.data.cpu().numpy())

        predictions = np.concatenate(predictions)

        return predictions

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(str(filename)))
