from models.pytorch_base_model import PyTorchBaseModel


class PyTorchMultiTargetBaseModel(PyTorchBaseModel):
    def __init__(self, **kwargs):
        super(PyTorchMultiTargetBaseModel, self).__init__(**kwargs)

        self.model = None
        self.model_source = None
        self.model_target = None

    def train_source(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        self.model = self.model_source
        super(PyTorchMultiTargetBaseModel, self).train(data_train, data_dev=data_dev, batch_size=batch_size,
                                                       epochs=epochs, verbose=verbose)
        self.model = None

    def train_target(self, data_train, data_dev=None, batch_size=512, epochs=10, verbose=1):
        self.model = self.model_target
        super(PyTorchMultiTargetBaseModel, self).train(data_train, data_dev=data_dev, batch_size=batch_size,
                                                       epochs=epochs, verbose=verbose)
