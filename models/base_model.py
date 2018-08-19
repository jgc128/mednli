import abc


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = None

    @abc.abstractmethod
    def build(self):
        """Build the model"""

    @abc.abstractmethod
    def train(self, data_train, data_dev=None, batch_size=512, epochs=10):
        """Train the model"""

    @abc.abstractmethod
    def load_weights(self, filename):
        """Load weights from the `filename`"""

    @abc.abstractmethod
    def evaluate(self, data, batch_size=512):
        """Evaluate the model on the provided data"""

