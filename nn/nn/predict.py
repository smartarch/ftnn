import tensorflow as tf


class NNPredictor():
    def __init__(self, weightsFile, shape):
        super().__init__()
        self.build(shape)
        self.load_weights(str(weightsFile))

    def predict(self, inputs):
        predictions = self(inputs, training=False)
        predIdxs = tf.argmax(predictions, axis=1).numpy()

        return predIdxs