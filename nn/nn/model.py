from tensorflow.keras import Model

class NNBase(Model):
    def __init__(self, **kwargs):
        super().__init__()
