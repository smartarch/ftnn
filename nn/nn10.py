import math

import tensorflow as tf
from tensorflow.keras import layers, activations

from nn.config import BalancedSimDataSetConfig, OnTheFlyMonteCarloSimDataSetConfig
from nn.config import TrainingConfig
from nn.model import NNBase
from nn.prep_otf import accessToWorkplaceDsGen
from nn.train import NNTrainable

'''
dsConfigJSONSim_90_10 = JSONSimDataSetConfig(
    name='ftnn-traces-v1',
    tracesName='ftnn-traces-v1',
    trainSimulationCount=90,
    valSimulationCount=10,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    outputSamplesPerSimulation=10800,
    allowedOnly=False
)

dsConfigOTF = OnTheFlyMonteCarloSimDataSetConfig(
    name='ftnn-otf',
    trainGenBatchCount=9,
    valGenBatchCount=1,
    allowedCountInBatch=500,
    deniedCountInBatch=500,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    batchGen=accessToWorkplaceDsGen
)
'''

def getDsBalancedOTFConfig(factor):
    return BalancedSimDataSetConfig(
        name='ftnn-otf',
        trainCount=9 * 48 * factor,
        valCount=1 * 48 * factor,
        inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
        outputKeys=['accessToWorkplace']
    )

def getDsBalancedCombinedConfig(factor):
    return BalancedSimDataSetConfig(
        name='ftnn-combined',
        trainCount=9 * 48 * factor,
        valCount=1 * 48 * factor,
        inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
        outputKeys=['accessToWorkplace']
    )

def getDsBalancedSynthConjConfig(factor, conjCount):
    deniedCatCount = 2 ** conjCount - 1

    return BalancedSimDataSetConfig(
        name='ftnn-synth-conj',
        trainCount=9 * deniedCatCount * 2 * factor,
        valCount=1 * deniedCatCount * 2 * factor,
        inputKeys=[f'y{idx}' for idx in range(conjCount)],
        outputKeys=['access']
    )



class DenseBlock(layers.Layer):
    def __init__(self, name, units, dropout=None, activation='relu'):
        super().__init__(name=name)

        self.dense = layers.Dense(name=f'{name}_dense', units=units)

        if activation == 'relu':
            self.activation = activations.relu
        elif activation == 'tanh':
            self.activation = activations.tanh
        elif activation == 'softmax':
            self.activation = activations.softmax
        elif activation == 'sigmoid':
            self.activation = activations.sigmoid

        if dropout is not None:
            self.dropout = layers.Dropout(name=f'{name}_dropout', rate=dropout)
        else:
            self.dropout = None

    def call(self, y, training=False):
        y = self.dense(y)
        y = self.activation(y)
        if self.dropout is not None:
            y = self.dropout(y, training=training)
        return y


class BelowThreshold(layers.Layer):
    def __init__(self, name, minValue, maxValue):
        super().__init__(name=name)
        self.offset = tf.constant(-minValue, dtype=tf.float32)
        self.scale = tf.constant(1 / (maxValue - minValue), dtype=tf.float32)
        self.threshold = tf.Variable(tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32), trainable=True)
        print(f'{name} ... {self.offset} {self.scale}')

    def call(self, y, training=False):
        y = (self.threshold - (y + self.offset) * self.scale) * 100
        y = activations.sigmoid(y)
        return y


class AboveThreshold(layers.Layer):
    def __init__(self, name, minValue, maxValue):
        super().__init__(name=name)
        self.offset = tf.constant(-minValue, dtype=tf.float32)
        self.scale = tf.constant(1 / (maxValue - minValue), dtype=tf.float32)
        self.threshold = tf.Variable(tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32), trainable=True)
        print(f'{name} ... {self.offset} {self.scale}')

    def call(self, y, training=False):
        y = ((y + self.offset) * self.scale - self.threshold) * 100
        y = activations.sigmoid(y)
        return y


class CorrectTime(layers.Layer):
    def __init__(self, name, minValue, maxValue, capacity, sigma=None, grid='random-static'):
        super().__init__(name=name)

        if sigma is None:
            sigma = (maxValue - minValue) / capacity

        if grid == 'equidistant-static':
            self.refPoints = tf.cast(tf.linspace(minValue, maxValue, capacity), dtype=tf.float32)
        elif grid == 'random-static':
            self.refPoints = tf.random.uniform(shape=(capacity,), minval=minValue, maxval=maxValue, dtype=tf.float32)
        elif grid == 'equidistant-trainable':
            self.refPoints = tf.Variable(
                tf.cast(tf.linspace(minValue, maxValue, capacity), dtype=tf.float32),
                trainable=True
            )
        elif grid == 'random-trainable':
            self.refPoints = tf.Variable(
                tf.random.uniform(shape=(capacity,), minval=minValue, maxval=maxValue, dtype=tf.float32),
                trainable=True
            )
        else:
            assert False

        self.dense = layers.Dense(name=f'{name}_dense', units=1)
        self.twiceSigmaSquare = tf.cast((2 * tf.square(sigma)), dtype=tf.float32)

    def call(self, y, training=False):
        y = tf.exp(-tf.square(self.refPoints - y) / self.twiceSigmaSquare)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class CorrectPlace(layers.Layer):
    def __init__(self, name, minPos, maxPos, capacity, sigma=None, grid='random-static'):
        super().__init__(name=name)

        numPoints = capacity ** 2

        if sigma is None:
            sigma = math.sqrt((maxPos[0] - minPos[0]) * (maxPos[1] - minPos[1]) / numPoints)

        if grid == 'random-static':
            self.refPoints = tf.stack(
                [tf.random.uniform(shape=(numPoints,), minval=minPos[0], maxval=maxPos[0], dtype=tf.float32), tf.random.uniform(shape=(numPoints,), minval=minPos[1], maxval=maxPos[1], dtype=tf.float32)],
                axis=1
            )
        elif grid == 'random-trainable':
            self.refPoints = tf.Variable(
                tf.stack(
                    [tf.random.uniform(shape=(numPoints,), minval=minPos[0], maxval=maxPos[0], dtype=tf.float32), tf.random.uniform(shape=(numPoints,), minval=minPos[1], maxval=maxPos[1], dtype=tf.float32)],
                    axis=1
                ),
                trainable=True
            )
        else:
            assert False

        self.dense = layers.Dense(name=f'{name}_dense', units=1)
        self.twiceSigmaSquare = tf.cast((2 * tf.square(sigma)), dtype=tf.float32)

    def call(self, y, training=False):
        y = tf.expand_dims(y, axis=1)
        y = tf.exp(-tf.reduce_sum(tf.square(self.refPoints - y), axis=2) / self.twiceSigmaSquare)
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class CorrectEvent(layers.Layer):
    def __init__(self, name, categoryCount, historyLength):
        super().__init__(name=name)
        self.categoryCount = categoryCount
        self.historyLength = historyLength
        self.dense = layers.Dense(name=f'{name}_dense', units=1)

    def call(self, y, training=False):
        y = self.dense(y)
        y = activations.sigmoid(y)
        return y


class NN10(NNBase):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        nnArch = config['nnArch']
        #self.timeLow = AboveThreshold('timeLow', minValue=0, maxValue=36000)
        #self.timeHigh = BelowThreshold('timeHigh', minValue=0, maxValue=36000)

        # self.time = CorrectTime('time', minValue=0, maxValue=36000, capacity=20)
        self.placeA = CorrectPlace('placeA', minPos=(0,0), maxPos=(316.43506,177.88289), capacity=20)
        self.placeB = CorrectPlace('placeB', minPos=(0,0), maxPos=(316.43506,177.88289), capacity=20)
        self.placeC = CorrectPlace('placeC', minPos=(0,0), maxPos=(316.43506,177.88289), capacity=20)
        self.headGear = CorrectEvent('headGear', categoryCount=2, historyLength=1)

        # self.point = [CorrectTime(name=f'points_{idx}', minValue=0, maxValue=1, capacity=20) for idx in range(10)]

    def call(self, y, training=False):
        #yTime = self.time(y[:,0:1], training=training)
        #yTimeLow = self.timeLow(y[:,0:1], training=training)
        #yTimeHigh = self.timeHigh(y[:,0:1], training=training)
        yTimeLow = tf.cast(y[:, 0:1] > 2400, dtype=tf.float32)
        yTimeHigh = tf.cast(y[:, 0:1] <= 33600, dtype=tf.float32)

        yPlaceA = self.placeA(y[:,1:3], training=training) * y[:,3:4]
        yPlaceB = self.placeB(y[:,1:3], training=training) * y[:,4:5]
        yPlaceC = self.placeC(y[:,1:3], training=training) * y[:,5:6]
        yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)
        #yPlace = yPlaceA + yPlaceB + yPlaceC

        yHeadGear = self.headGear(y[:,6:8], training=training)

        y = activations.sigmoid((yTimeLow + yTimeHigh + yPlace + yHeadGear - 3.5) * 10)
        #y = yTime * yPlace * yHeadGear

        # yPoint0 = self.point[0](y[:, 0:1], training=training)
        # yPoint1 = self.point[1](y[:, 1:2], training=training)
        # yPoint2 = self.point[2](y[:, 2:3], training=training)
        # yPoint3 = self.point[3](y[:, 3:4], training=training)
        # yPoint4 = self.point[4](y[:, 4:5], training=training)
        # yPoint5 = self.point[5](y[:, 5:6], training=training)
        # yPoint6 = self.point[6](y[:, 6:7], training=training)
        # yPoint7 = self.point[7](y[:, 7:8], training=training)
        # yPoint8 = self.point[8](y[:, 8:9], training=training)
        # yPoint9 = self.point[9](y[:, 9:10], training=training)
        # y = activations.sigmoid((yPoint0 + yPoint1 + yPoint2 + yPoint3 + yPoint4 + yPoint5 + yPoint6 + yPoint7 + yPoint8 + yPoint9 - 9.5) * 10000)

        return y


class TrainableNN10(NNTrainable, NN10):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def createConfig(cls, config):
        return TrainingConfig(
            #dsConfig=getDsBalancedSynthConjConfig(100, 10),
            #dsConfig=getDsBalancedCombinedConfig(100),
            dsConfig=getDsBalancedOTFConfig(100),
            # dsConfig=OnTheFlyMonteCarloSimDataSetConfig(
            #     name='ftnn-otf',
            #     trainGenBatchCount=9,
            #     valGenBatchCount=1,
            #     allowedCountInBatch=5000,
            #     deniedCountInBatch=5000,
            #     inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
            #     outputKeys=['accessToWorkplace'],
            #     batchGen=accessToWorkplaceDsGen
            # ),
            batchSize=config['batchSize'],
            learningRate=config['learningRate'],
            nnArch=config['nnArch']
        )


