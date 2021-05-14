#!/usr/bin/env python3
import argparse
import math
import h5py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations


def loadData(fileName, batchSize):
    with h5py.File(fileName, 'r') as hf:
        trainDS = tf.data.Dataset.from_tensor_slices((hf['train-inputs'], hf['train-outputs'])).cache()
        valDS = tf.data.Dataset.from_tensor_slices((hf['val-inputs'], hf['val-outputs'])).cache()

    trainDS = trainDS.shuffle(buffer_size=len(trainDS)).batch(batchSize).prefetch(1000)
    valDS = valDS.batch(batchSize).prefetch(1000)

    return (trainDS, valDS)


class CorrectTime(layers.Layer):
    def __init__(self, name, minValue, maxValue, capacity, sigma=None):
        super().__init__(name=name)

        if sigma is None:
            sigma = (maxValue - minValue) / capacity

        # random static points
        self.refPoints = tf.random.uniform(shape=(capacity,), minval=minValue, maxval=maxValue, dtype=tf.float32)

        self.dense = layers.Dense(name=f'{name}_dense', units=1)
        self.twiceSigmaSquare = tf.cast((2 * tf.square(sigma)), dtype=tf.float32)

    def call(self, y, training=False):
        y = tf.exp(-tf.square(self.refPoints - y) / self.twiceSigmaSquare)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class CorrectPlace(layers.Layer):
    def __init__(self, name, minPos, maxPos, capacity, sigma=None):
        super().__init__(name=name)

        numPoints = capacity ** 2

        if sigma is None:
            sigma = math.sqrt((maxPos[0] - minPos[0]) * (maxPos[1] - minPos[1]) / numPoints)

        self.refPoints = tf.stack(
            [
                tf.random.uniform(shape=(numPoints,), minval=minPos[0], maxval=maxPos[0], dtype=tf.float32),
                tf.random.uniform(shape=(numPoints,), minval=minPos[1], maxval=maxPos[1], dtype=tf.float32)
            ],
            axis=1
        )

        self.dense = layers.Dense(name=f'{name}_dense', units=1)
        self.twiceSigmaSquare = tf.cast((2 * tf.square(sigma)), dtype=tf.float32)

    def call(self, y):
        y = tf.expand_dims(y, axis=1)
        y = tf.exp(-tf.reduce_sum(tf.square(self.refPoints - y), axis=2) / self.twiceSigmaSquare)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class CorrectEvent(layers.Layer):
    def __init__(self, name, capacity):
        super().__init__(name=name)
        self.dense = layers.Dense(name=f'{name}_dense_last', units=1)
        self.hidden = None
        if capacity > 1:
            self.hidden = layers.Dense(name=f'{name}_dense_hidden', units=capacity, activation='relu')

    def call(self, y):
        if self.hidden is not None:
            y = self.hidden(y)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class ModelI0(tf.keras.Model):
    def __init__(self, args):
        super(ModelI0, self).__init__()
        self.fromTime = 2400
        self.toTime = 33600
        self.gatePositions = [
            [178.64561, 98.856514],
            [237.03545, 68.872505],
            [237.0766, 135.65627],
        ]

    def atGate(self, id, x, y):
        gatePos = self.gatePositions[id]
        dx = x - gatePos[0]
        dy = y - gatePos[1]
        return tf.math.sqrt(dx * dx + dy * dy) <= 10

    def call(self, y):
        yTimeLow = tf.cast(y[:, 0] > self.fromTime, dtype=tf.float32)
        yTimeHigh = tf.cast(y[:, 0] <= self.toTime, dtype=tf.float32)

        yPlaceA = tf.cast(tf.logical_and(self.atGate(0, y[:, 1], y[:, 2]), y[:, 3] == 1), dtype=tf.float32)
        yPlaceB = tf.cast(tf.logical_and(self.atGate(1, y[:, 1], y[:, 2]), y[:, 4] == 1), dtype=tf.float32)
        yPlaceC = tf.cast(tf.logical_and(self.atGate(2, y[:, 1], y[:, 2]), y[:, 5] == 1), dtype=tf.float32)
        yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)  # OR

        yHeadGear = tf.cast(tf.logical_and(y[:, 6] == 1, y[:, 7] == 0), dtype=tf.float32)
        return activations.sigmoid((yTimeLow + yTimeHigh + yPlace + yHeadGear - 3.5) * 10)  # AND


class ModelI2(tf.keras.Model):
    def __init__(self, args):
        super(ModelI2, self).__init__()
        self.time = CorrectTime('time', minValue=0, maxValue=36000, capacity=args.time_capacity)
        self.placeA = CorrectPlace('placeA', minPos=(0, 0), maxPos=(316.43506, 177.88289), capacity=args.place_capacity)
        self.placeB = CorrectPlace('placeB', minPos=(0, 0), maxPos=(316.43506, 177.88289), capacity=args.place_capacity)
        self.placeC = CorrectPlace('placeC', minPos=(0, 0), maxPos=(316.43506, 177.88289), capacity=args.place_capacity)
        self.headGear = CorrectEvent('headGear', capacity=args.event_capacity)

    def call(self, y):
        yTime = self.time(y[:, 0:1])
        yPlaceA = self.placeA(y[:, 1:3]) * y[:, 3:4]
        yPlaceB = self.placeB(y[:, 1:3]) * y[:, 4:5]
        yPlaceC = self.placeC(y[:, 1:3]) * y[:, 5:6]
        yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)  # OR
        yHeadGear = self.headGear(y[:, 6:8])

        return activations.sigmoid((yTime + yPlace + yHeadGear - 2.5) * 10)  # AND


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file path")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate")
    parser.add_argument("--learning_rate_decay", default=True, action="store_true", help="Decay learning rate")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--time_capacity", default=20, type=int, help="Learning capacity of time layers.")
    parser.add_argument("--place_capacity", default=20, type=int, help="Learning capacity of place layers.")
    parser.add_argument("--event_capacity", default=20, type=int, help="Learning capacity of event layers.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Fitting and evaluation will be verbose.")
    args = parser.parse_args()

    # Use given number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    trainDS, valDS = loadData(args.input_file, args.batch_size)

    model = ModelI0(args)

    learning_rate = args.learning_rate
    if args.learning_rate_decay:
        learning_rate = tf.keras.experimental.CosineDecay(learning_rate, args.epochs * len(trainDS))
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    if args.verbose:
        print("File: {}, training size: {}, validation size: {}".format(
            args.input_file, len(trainDS), len(valDS)))
        print("Batch size: {}, epochs: {}, learning rate: {}, learning decay: {}, label smoothing: {}".format(
            args.batch_size, args.epochs, args.learning_rate, args.learning_rate_decay, args.label_smoothing))
        model.build(input_shape=trainDS.element_spec[0].shape)
        model.summary()

    accuracies = []
    for epochIdx in range(0, args.epochs):
        model.fit(trainDS, epochs=1, verbose=args.verbose)
        evalRes = model.evaluate(valDS, return_dict=True, verbose=args.verbose)
        accuracies.append(evalRes['accuracy'])

    # TODO fix with models
    # bareFileName = re.sub(r"^.*/", "", args.input_file)
    # print("{};dense-{};{}".format(bareFileName, "-".join(sys.argv[2:]), ";".join(map(str, accuracies))))
