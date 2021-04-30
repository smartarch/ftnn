from tensorflow.keras import layers, activations

from nn.config import JSONSimDataSetConfig, OnTheFlySimDataSetConfig
from nn.config import TrainingConfig
from nn.model import NNBase
from nn.train import NNTrainable

import tensorflow as tf
import tensorflow_probability as tfp

def timeOTFDsGen(dsConfig):
    genBatchSize = dsConfig['genBatchSize']
    genSize = genBatchSize * 1000
    gatePosByWpId = tf.constant([[178.64561, 98.856514], [237.03545, 68.872505], [237.0766, 135.65627]], dtype=tf.float32)
    gateAccessPosDistanceThreshold = 5

    time = tf.random.uniform(shape=(genSize,), minval=0, maxval=36000, dtype=tf.float32)
    posX = tf.random.uniform(shape=(genSize,), minval=0, maxval=316.43506, dtype=tf.float32)
    posY = tf.random.uniform(shape=(genSize,), minval=0, maxval=177.88289, dtype=tf.float32)
    pos = tf.stack([posX, posY], axis=1)

    wpId = tf.random.uniform(shape=(genSize,), minval=0, maxval=3, dtype=tf.int32)
    wpId_OneHot = tf.one_hot(wpId, depth=3, dtype=tf.float32)

    gatePos = tf.gather(gatePosByWpId, wpId, axis=0)
    gateDistSquaredElems = tf.math.square(pos - gatePos)
    gateDist = tf.math.sqrt(gateDistSquaredElems[:,0] + gateDistSquaredElems[:,1])

    hgerEvents_0 = tf.random.uniform(shape=(genSize,), minval=0, maxval=2, dtype=tf.int32)
    hgerEvents_0_OneHot = tf.one_hot(hgerEvents_0, depth=2, dtype=tf.float32)

    inputs = tf.concat([tf.expand_dims(time, axis=1), pos, wpId_OneHot, hgerEvents_0_OneHot], axis=1)

    isAccessRequested = gateDist <= gateAccessPosDistanceThreshold
    atGate = gateDist <= 10
    hasHeadGear = hgerEvents_0 == 1
    isRightTime = tf.logical_and(time >= 2400, time <= 33600)

    outputs = tf.reduce_all([isAccessRequested, atGate, hasHeadGear, isRightTime], axis=0)

    deniedAccesses = outputs == False
    allowedAccesses = outputs == True

    deniedAccessCount = tf.reduce_sum(tf.cast(deniedAccesses, dtype=tf.int32))
    allowedAccessCount = tf.reduce_sum(tf.cast(allowedAccesses, dtype=tf.int32))

    deniedWeight = 0.5 * tf.cast(1 / deniedAccessCount, dtype=tf.float32) if deniedAccessCount > 0 else tf.constant(0, dtype=tf.float32)
    allowedWeight = 0.5 * tf.cast(1 / allowedAccessCount, dtype=tf.float32) if allowedAccessCount > 0 else tf.constant(0, dtype=tf.float32)
    probByOutput = tf.stack([deniedWeight, allowedWeight], axis=0)

    outputs = tf.cast(outputs, dtype=tf.int32)

    probs = tf.gather(probByOutput, outputs, axis=0)
    selDist = tfp.distributions.Categorical(probs=probs)
    sel = selDist.sample((genBatchSize,))

    inputs = tf.gather(inputs, sel, axis=0)
    outputs = tf.expand_dims(tf.cast(outputs, dtype=tf.float32), axis=1)
    outputs = tf.gather(outputs, sel, axis=0)
    weights = tf.ones(shape=(genBatchSize, 1), dtype=tf.float32)

    return (inputs, outputs, weights)


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

dsConfigJSONSim_900_100 = JSONSimDataSetConfig(
    name='ftnn-traces-v1',
    tracesName='ftnn-traces-v1',
    trainSimulationCount=900,
    valSimulationCount=100,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    outputSamplesPerSimulation=10800,
    allowedOnly = False
)

dsConfigOTF1 = OnTheFlySimDataSetConfig(
    name='ftnn-otf-1',
    trainGenBatchCount=9,
    valGenBatchCount=1,
    genBatchSize=1000,
    inputKeys=['time', 'posX', 'posY', 'wpId:A', 'wpId:B', 'wpId:C', 'hgerEvents[0]:TAKE_HGEAR', 'hgerEvents[0]:RET_HGEAR'],
    outputKeys=['accessToWorkplace'],
    deniedOnly=False
)
dsConfigOTF1['batchGen'] = lambda genBatchIdx: timeOTFDsGen(dsConfigOTF1)

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


class CorrectTime(layers.Layer):
    def __init__(self, name, minValue, maxValue, numPoints, sigma, grid='random'):
        super().__init__(name=name)
        if grid == 'equidistant':
            self.refPoints = tf.cast(tf.linspace(minValue, maxValue, numPoints), dtype=tf.float32)
        elif grid == 'random':
            self.refPoints = tf.random.uniform(shape=(numPoints,), minval=minValue, maxval=maxValue, dtype=tf.float32)
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
    def __init__(self, name, minPos, maxPos, numPoints, sigma, grid='random'):
        super().__init__(name=name)
        if grid == 'random':
            self.refPoints = tf.stack(
                [tf.random.uniform(shape=(numPoints,), minval=minPos[0], maxval=maxPos[0], dtype=tf.float32), tf.random.uniform(shape=(numPoints,), minval=minPos[1], maxval=maxPos[1], dtype=tf.float32)],
                axis=1
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
        self.time = CorrectTime('time', minValue=0, maxValue=36000, numPoints=2000, sigma=30)
        self.placeA = CorrectPlace('placeA', minPos=(0,0), maxPos=(316.43506,177.88289), numPoints=8000, sigma=5)
        self.placeB = CorrectPlace('placeB', minPos=(0,0), maxPos=(316.43506,177.88289), numPoints=8000, sigma=5)
        self.placeC = CorrectPlace('placeC', minPos=(0,0), maxPos=(316.43506,177.88289), numPoints=8000, sigma=5)
        self.headGear = CorrectEvent('headGear', categoryCount=2, historyLength=1)

    def call(self, y, training=False):
        yTime = self.time(y[:,0:1], training=training)

        yPlaceA = self.placeA(y[:,1:3], training=training) * y[:,3:4]
        yPlaceB = self.placeB(y[:,1:3], training=training) * y[:,4:5]
        yPlaceC = self.placeC(y[:,1:3], training=training) * y[:,5:6]
        yPlace = yPlaceA + yPlaceB + yPlaceC

        yHeadGear = self.headGear(y[:,6:8], training=training)

        y = yTime * yPlace * yHeadGear

        return y


class TrainableNN10(NNTrainable, NN10):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def createConfig(cls, config):
        return TrainingConfig(
            dsConfig=dsConfigOTF1,
            batchSize=config['batchSize'],
            learningRate=config['learningRate'],
            nnArch=config['nnArch']
        )


