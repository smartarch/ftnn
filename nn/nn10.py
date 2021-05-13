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

        self.time = CorrectTime('time', minValue=0, maxValue=36000, capacity=20)
        self.placeA = CorrectPlace('placeA', minPos=(0,0), maxPos=(316.43506,177.88289), capacity=20)
        self.placeB = CorrectPlace('placeB', minPos=(0,0), maxPos=(316.43506,177.88289), capacity=20)
        self.placeC = CorrectPlace('placeC', minPos=(0,0), maxPos=(316.43506,177.88289), capacity=20)
        self.headGear = CorrectEvent('headGear', categoryCount=2, historyLength=1)

        # self.point = [CorrectTime(name=f'points_{idx}', minValue=0, maxValue=1, capacity=20) for idx in range(10)]

    def call(self, y, training=False):
        '''
        varianta I0: OTF a combined
        vsechno je striktne = nic se neuci - jen fuzifikace

        yTimeLow = tf.cast(y[:, 0:1] > 2400, dtype=tf.float32)
        yTimeHigh = tf.cast(y[:, 0:1] <= 33600, dtype=tf.float32)

        ... zmenit na to, aby byla strict check na pozici
        ... yPlaceA = self.placeA(y[:,1:3], training=training) * y[:,3:4]
        ... yPlaceB = self.placeB(y[:,1:3], training=training) * y[:,4:5]
        ... yPlaceC = self.placeC(y[:,1:3], training=training) * y[:,5:6]
        ... yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)

        yHeadGear = tf.cast(tf.logical_and(y[:,6] == 1, y[:,7] == 0), dtype=tf.float32)

        y = activations.sigmoid((yTimeLow + yTimeHigh + yPlace + yHeadGear - 3.5) * 10)

        varianta I1: OTF a combined
        cas se uci intervalem

        yTimeLow = self.timeLow(y[:,0:1], training=training)
        yTimeHigh = self.timeHigh(y[:,0:1], training=training)

        ... zmenit na to, aby byla strict check na pozici
        ... yPlaceA = self.placeA(y[:,1:3], training=training) * y[:,3:4]
        ... yPlaceB = self.placeB(y[:,1:3], training=training) * y[:,4:5]
        ... yPlaceC = self.placeC(y[:,1:3], training=training) * y[:,5:6]
        ... yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)

        yHeadGear = tf.cast(tf.logical_and(y[:,6] == 1, y[:,7] == 0), dtype=tf.float32)

        y = activations.sigmoid((yTimeLow + yTimeHigh + yPlace + yHeadGear - 3.5) * 10)


        varianta I2: OTF a combined
        vsechno se uci
        yTime = self.time(y[:,0:1], training=training)

        yPlaceA = self.placeA(y[:,1:3], training=training) * y[:,3:4]
        yPlaceB = self.placeB(y[:,1:3], training=training) * y[:,4:5]
        yPlaceC = self.placeC(y[:,1:3], training=training) * y[:,5:6]
        yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)

        yHeadGear = self.headGear(y[:,6:8], training=training)

        y = activations.sigmoid((yTime + yPlace + yHeadGear - 2.5) * 10)


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
time (CorrectTime)           multiple                  21
_________________________________________________________________
placeA (CorrectPlace)        multiple                  401
_________________________________________________________________
placeB (CorrectPlace)        multiple                  401
_________________________________________________________________
placeC (CorrectPlace)        multiple                  401
_________________________________________________________________
headGear (CorrectEvent)      multiple                  3
=================================================================
Total params: 1,227
Trainable params: 1,227
Non-trainable params: 0
_________________________________________________________________
2021-05-13 12:47:14.246210: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-05-13 12:47:14.262292: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2100000000 Hz
2021-05-13 12:47:14.950818: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-05-13 12:47:15.677272: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-05-13 12:47:16,773: Epoch 1: loss = 0.3307928442955017, accuracy = 90.77314758300781, val loss = 0.06448422372341156, val accuracy = 99.375, 6.040478599142041 ms/batch
2021-05-13 12:47:17,414: Epoch 2: loss = 0.04786796122789383, accuracy = 99.60185241699219, val loss = 0.04176493361592293, val accuracy = 99.54166412353516, 1.362208078598835 ms/batch
2021-05-13 12:47:18,054: Epoch 3: loss = 0.03448233753442764, accuracy = 99.68286895751953, val loss = 0.03368227556347847, val accuracy = 99.58333587646484, 1.359432406679413 ms/batch
2021-05-13 12:47:18,686: Epoch 4: loss = 0.028433116152882576, accuracy = 99.73379516601562, val loss = 0.02845649980008602, val accuracy = 99.66666412353516, 1.3430584111862633 ms/batch
2021-05-13 12:47:19,327: Epoch 5: loss = 0.024217944592237473, accuracy = 99.77314758300781, val loss = 0.025004805997014046, val accuracy = 99.72917175292969, 1.362805535807412 ms/batch
2021-05-13 12:47:19,959: Epoch 6: loss = 0.021054640412330627, accuracy = 99.80786895751953, val loss = 0.02304246462881565, val accuracy = 99.72917175292969, 1.3409909412000307 ms/batch
2021-05-13 12:47:20,592: Epoch 7: loss = 0.01901744119822979, accuracy = 99.84259033203125, val loss = 0.021284841001033783, val accuracy = 99.79166412353516, 1.3410487823937771 ms/batch
2021-05-13 12:47:21,228: Epoch 8: loss = 0.01766192726790905, accuracy = 99.83564758300781, val loss = 0.01992999203503132, val accuracy = 99.72917175292969, 1.3431099039563061 ms/batch
2021-05-13 12:47:21,858: Epoch 9: loss = 0.01655133068561554, accuracy = 99.86573791503906, val loss = 0.01883961446583271, val accuracy = 99.79166412353516, 1.3409429753320457 ms/batch
2021-05-13 12:47:22,490: Epoch 10: loss = 0.01573871448636055, accuracy = 99.8611068725586, val loss = 0.01807679980993271, val accuracy = 99.79166412353516, 1.352522500167937 ms/batch
2021-05-13 12:47:23,119: Epoch 11: loss = 0.014980529434978962, accuracy = 99.8865737915039, val loss = 0.01755637302994728, val accuracy = 99.79166412353516, 1.3392338385948768 ms/batch
2021-05-13 12:47:23,771: Epoch 12: loss = 0.014386632479727268, accuracy = 99.89814758300781, val loss = 0.016913890838623047, val accuracy = 99.77082824707031, 1.3895112381884331 ms/batch
2021-05-13 12:47:24,404: Epoch 13: loss = 0.013940745033323765, accuracy = 99.8773193359375, val loss = 0.01665785163640976, val accuracy = 99.8125, 1.3428369217370388 ms/batch
2021-05-13 12:47:25,037: Epoch 14: loss = 0.013508522883057594, accuracy = 99.89814758300781, val loss = 0.016315363347530365, val accuracy = 99.79166412353516, 1.3449389553634372 ms/batch
2021-05-13 12:47:25,656: Epoch 15: loss = 0.01315925270318985, accuracy = 99.88426208496094, val loss = 0.016009613871574402, val accuracy = 99.75, 1.3105192128017809 ms/batch
2021-05-13 12:47:26,291: Epoch 16: loss = 0.012846358120441437, accuracy = 99.91203308105469, val loss = 0.015592682175338268, val accuracy = 99.77082824707031, 1.3577973348854562 ms/batch
2021-05-13 12:47:26,934: Epoch 17: loss = 0.012553078122437, accuracy = 99.89583587646484, val loss = 0.01479197945445776, val accuracy = 99.79166412353516, 1.3706197400064863 ms/batch
2021-05-13 12:47:27,559: Epoch 18: loss = 0.01215505599975586, accuracy = 99.90972137451172, val loss = 0.014591234736144543, val accuracy = 99.79166412353516, 1.3184413402038213 ms/batch
2021-05-13 12:47:28,188: Epoch 19: loss = 0.01188491377979517, accuracy = 99.90972137451172, val loss = 0.013749822042882442, val accuracy = 99.79166412353516, 1.3407257181652905 ms/batch
2021-05-13 12:47:28,825: Epoch 20: loss = 0.011583426967263222, accuracy = 99.91435241699219, val loss = 0.013469354249536991, val accuracy = 99.79166412353516, 1.3474106083254842 ms/batch
2021-05-13 12:47:29,482: Epoch 21: loss = 0.011376245878636837, accuracy = 99.91435241699219, val loss = 0.012649621814489365, val accuracy = 99.8125, 1.4111403177475788 ms/batch
2021-05-13 12:47:30,110: Epoch 22: loss = 0.01115580927580595, accuracy = 99.91203308105469, val loss = 0.012212724424898624, val accuracy = 99.89583587646484, 1.3434915147589508 ms/batch
2021-05-13 12:47:30,742: Epoch 23: loss = 0.011014312505722046, accuracy = 99.9282455444336, val loss = 0.011725498363375664, val accuracy = 99.83333587646484, 1.3530868044971713 ms/batch
2021-05-13 12:47:31,372: Epoch 24: loss = 0.01074885856360197, accuracy = 99.9259262084961, val loss = 0.011945885606110096, val accuracy = 99.83333587646484, 1.3450299494365263 ms/batch
2021-05-13 12:47:32,014: Epoch 25: loss = 0.010664275847375393, accuracy = 99.9282455444336, val loss = 0.011081141419708729, val accuracy = 99.89583587646484, 1.362887359935151 ms/batch
2021-05-13 12:47:32,647: Epoch 26: loss = 0.010458243079483509, accuracy = 99.93287658691406, val loss = 0.01095041073858738, val accuracy = 99.875, 1.3449340177005564 ms/batch
2021-05-13 12:47:33,284: Epoch 27: loss = 0.01032894104719162, accuracy = 99.92361450195312, val loss = 0.010773354209959507, val accuracy = 99.85416412353516, 1.3600369176921054 ms/batch
2021-05-13 12:47:33,926: Epoch 28: loss = 0.01021508313715458, accuracy = 99.92361450195312, val loss = 0.010509861633181572, val accuracy = 99.91666412353516, 1.368959274517714 ms/batch
2021-05-13 12:47:34,568: Epoch 29: loss = 0.010115224868059158, accuracy = 99.9398193359375, val loss = 0.0105606559664011, val accuracy = 99.875, 1.3648560766637678 ms/batch
2021-05-13 12:47:35,208: Epoch 30: loss = 0.010041151195764542, accuracy = 99.93518829345703, val loss = 0.010428173467516899, val accuracy = 99.9375, 1.3663789929722894 ms/batch
2021-05-13 12:47:35,836: Epoch 31: loss = 0.009950868785381317, accuracy = 99.9375, val loss = 0.010259164497256279, val accuracy = 99.875, 1.3440614621314777 ms/batch
2021-05-13 12:47:36,474: Epoch 32: loss = 0.00987636111676693, accuracy = 99.9398193359375, val loss = 0.010274169035255909, val accuracy = 99.9375, 1.3667309777976493 ms/batch
2021-05-13 12:47:37,117: Epoch 33: loss = 0.009845119901001453, accuracy = 99.94444274902344, val loss = 0.010167412459850311, val accuracy = 99.89583587646484, 1.373250808941542 ms/batch
2021-05-13 12:47:37,746: Epoch 34: loss = 0.009777923114597797, accuracy = 99.94676208496094, val loss = 0.010118396021425724, val accuracy = 99.9375, 1.3398552787374463 ms/batch
2021-05-13 12:47:38,383: Epoch 35: loss = 0.009742255322635174, accuracy = 99.93518829345703, val loss = 0.010126633569598198, val accuracy = 99.95832824707031, 1.358186704872628 ms/batch
2021-05-13 12:47:39,017: Epoch 36: loss = 0.009697448462247849, accuracy = 99.95370483398438, val loss = 0.010053874924778938, val accuracy = 99.91666412353516, 1.3582332599797897 ms/batch
2021-05-13 12:47:39,660: Epoch 37: loss = 0.009653481654822826, accuracy = 99.95138549804688, val loss = 0.010017204098403454, val accuracy = 99.91666412353516, 1.364073104406955 ms/batch
2021-05-13 12:47:40,296: Epoch 38: loss = 0.009622142650187016, accuracy = 99.95370483398438, val loss = 0.009992234408855438, val accuracy = 99.91666412353516, 1.3629064052062627 ms/batch
2021-05-13 12:47:40,933: Epoch 39: loss = 0.009598108008503914, accuracy = 99.95138549804688, val loss = 0.009967753663659096, val accuracy = 99.9375, 1.3580315211820884 ms/batch
2021-05-13 12:47:41,567: Epoch 40: loss = 0.00956665351986885, accuracy = 99.95832824707031, val loss = 0.009953497909009457, val accuracy = 99.9375, 1.353961476207485 ms/batch
2021-05-13 12:47:42,198: Epoch 41: loss = 0.009546425193548203, accuracy = 99.95601654052734, val loss = 0.009934047237038612, val accuracy = 99.9375, 1.3538260431684686 ms/batch
2021-05-13 12:47:42,818: Epoch 42: loss = 0.009528324007987976, accuracy = 99.95601654052734, val loss = 0.009929132647812366, val accuracy = 99.9375, 1.3206442432290704 ms/batch
2021-05-13 12:47:43,452: Epoch 43: loss = 0.009511963464319706, accuracy = 99.95601654052734, val loss = 0.009918111376464367, val accuracy = 99.91666412353516, 1.3536729756191637 ms/batch
2021-05-13 12:47:44,092: Epoch 44: loss = 0.009503263048827648, accuracy = 99.95370483398438, val loss = 0.009915578179061413, val accuracy = 99.91666412353516, 1.3567385588877299 ms/batch
2021-05-13 12:47:44,735: Epoch 45: loss = 0.009490828029811382, accuracy = 99.95370483398438, val loss = 0.009909836575388908, val accuracy = 99.91666412353516, 1.3542690220669176 ms/batch
2021-05-13 12:47:45,382: Epoch 46: loss = 0.009484481066465378, accuracy = 99.95601654052734, val loss = 0.009906772524118423, val accuracy = 99.9375, 1.3773970350005922 ms/batch
2021-05-13 12:47:46,027: Epoch 47: loss = 0.009478633292019367, accuracy = 99.95601654052734, val loss = 0.009904979728162289, val accuracy = 99.91666412353516, 1.3775677370601858 ms/batch
2021-05-13 12:47:46,665: Epoch 48: loss = 0.00947466492652893, accuracy = 99.95601654052734, val loss = 0.009904774837195873, val accuracy = 99.9375, 1.3592539454352925 ms/batch
2021-05-13 12:47:47,296: Epoch 49: loss = 0.009472791105508804, accuracy = 99.95601654052734, val loss = 0.009904004633426666, val accuracy = 99.91666412353516, 1.3519920540984565 ms/batch
2021-05-13 12:47:47,949: Epoch 50: loss = 0.009471625089645386, accuracy = 99.95601654052734, val loss = 0.0099038640037179, val accuracy = 99.91666412353516, 1.391220374925602 ms/batch
2021-05-13 12:47:47,949: Total number of batches: 16900








        varianta C5: synth5
        yPoint0 = self.point[0](y[:, 0:1], training=training)
        yPoint1 = self.point[1](y[:, 1:2], training=training)
        yPoint2 = self.point[2](y[:, 2:3], training=training)
        yPoint3 = self.point[3](y[:, 3:4], training=training)
        yPoint4 = self.point[4](y[:, 4:5], training=training)

        y = activations.sigmoid((yPoint0 + yPoint1 + yPoint2 + yPoint3 + yPoint4 - 4.5) * 10)


        varianta C10: synth10
        yPoint0 = self.point[0](y[:, 0:1], training=training)
        yPoint1 = self.point[1](y[:, 1:2], training=training)
        yPoint2 = self.point[2](y[:, 2:3], training=training)
        yPoint3 = self.point[3](y[:, 3:4], training=training)
        yPoint4 = self.point[4](y[:, 4:5], training=training)
        yPoint5 = self.point[5](y[:, 5:6], training=training)
        yPoint6 = self.point[6](y[:, 6:7], training=training)
        yPoint7 = self.point[7](y[:, 7:8], training=training)
        yPoint8 = self.point[8](y[:, 8:9], training=training)
        yPoint9 = self.point[9](y[:, 9:10], training=training)

        y = activations.sigmoid((yPoint0 + yPoint1 + yPoint2 + yPoint3 + yPoint4 + yPoint5 + yPoint6 + yPoint7 + yPoint8 + yPoint9 - 9.5) * 10)
        '''




        yTime = self.time(y[:,0:1], training=training)
        #yTimeLow = self.timeLow(y[:,0:1], training=training)
        #yTimeHigh = self.timeHigh(y[:,0:1], training=training)
        # yTimeLow = tf.cast(y[:, 0:1] > 2400, dtype=tf.float32)
        # yTimeHigh = tf.cast(y[:, 0:1] <= 33600, dtype=tf.float32)

        yPlaceA = self.placeA(y[:,1:3], training=training) * y[:,3:4]
        yPlaceB = self.placeB(y[:,1:3], training=training) * y[:,4:5]
        yPlaceC = self.placeC(y[:,1:3], training=training) * y[:,5:6]
        yPlace = activations.sigmoid((yPlaceA + yPlaceB + yPlaceC - 0.5) * 10)
        #yPlace = yPlaceA + yPlaceB + yPlaceC

        #yHeadGear = tf.cast(tf.logical_and(y[:,6] == 1, y[:,7] == 0), dtype=tf.float32)
        yHeadGear = self.headGear(y[:,6:8], training=training)

        y = activations.sigmoid((yTime + yPlace + yHeadGear - 2.5) * 10)
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


