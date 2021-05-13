import logging
from time import time

import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics

from nn.dataset import loadDS
from nn.model import NNBase


class NNTrainable(NNBase):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        self.config = config

        trainDS, valDS = loadDS(config['dsConfig'], config['batchSize'])
        self.trainDS = trainDS
        self.valDS = valDS


    def buildFromConfig(self):
        self.build(input_shape=self.trainDS.element_spec[0].shape)


    @tf.function
    def trainStep(self, inputs, outputs, weights, optimizer, lossFn, trainLoss, trainAccuracy):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)

            loss = lossFn(outputs, predictions, weights)
            loss += sum(self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        trainLoss(loss)
        trainAccuracy.update_state(outputs, predictions, weights)


    @tf.function
    def evalStep(self, inputs, outputs, weights, lossFn, valLoss, valAccuracy):
        predictions = self(inputs, training=False)

        loss = lossFn(outputs, predictions, weights)
        loss += sum(self.losses)

        valLoss(loss)
        valAccuracy.update_state(outputs, predictions, weights)


    def train(self, epochsTotal, yieldFn = None):
        trainDS = self.trainDS
        valDS = self.valDS

        learningRate = self.config['learningRate']
        # for getDsBalancedSynthConjConfig(100, 10) ... learningRate = tf.keras.experimental.CosineDecay(learningRate, (2 ** 10 - 1) * 2 * 9 * 100 * 50 // 128)
        # for getDsBalancedOTFConfig(1000) ... learningRate = tf.keras.experimental.CosineDecay(learningRate, 9 * 48 * 1000 * 50 // 128)
        learningRate = tf.keras.experimental.CosineDecay(learningRate, 9 * 48 * 100 * 50 // 128)
        optimizer = optimizers.Adam(learning_rate=learningRate)
        step = tf.Variable(0, trainable=False, dtype=tf.int32)

        lossFn = losses.BinaryCrossentropy()
        trainLoss = metrics.Mean(name='train_loss')
        trainAccuracy = metrics.BinaryAccuracy(name='train_accuracy')
        valLoss = metrics.Mean(name='val_loss')
        valAccuracy = metrics.BinaryAccuracy(name='val_accuracy')

        totalBatches = 0
        while step.numpy() < epochsTotal:
            step.assign_add(1)
            epoch = step.numpy()

            trainLoss.reset_states()
            trainAccuracy.reset_states()
            valLoss.reset_states()
            valAccuracy.reset_states()

            cumulativeTime = 0
            cumulativeTimeMeasurements = 0

            for batchIdx, (inputs, outputs, weights) in enumerate(trainDS):
                startTime = time()

                self.trainStep(inputs, outputs, weights, optimizer, lossFn, trainLoss, trainAccuracy)

                endTime = time()
                cumulativeTimeMeasurements += 1
                cumulativeTime += endTime - startTime

                if (batchIdx + 1) % 1000 == 0:
                    logging.info(f'Epoch {epoch}: training {batchIdx + 1}, {cumulativeTime * 1000 / cumulativeTimeMeasurements} ms/batch')

                    if yieldFn is not None:
                        yieldFn()

                totalBatches += 1

            for batchIdx, (inputs, outputs, weights) in enumerate(valDS):
                self.evalStep(inputs, outputs, weights, lossFn, valLoss, valAccuracy)

                if (batchIdx + 1) % 1000 == 0:
                    logging.info(f'Epoch {epoch}: validating {batchIdx + 1}')

                    if yieldFn is not None:
                        yieldFn()

            logging.info(f'Epoch {epoch}: loss = {trainLoss.result()}, accuracy = {trainAccuracy.result() * 100}, val loss = {valLoss.result()}, val accuracy = {valAccuracy.result() * 100}, {cumulativeTime * 1000 / cumulativeTimeMeasurements} ms/batch')

            if yieldFn is not None:
                yieldFn()

        logging.info(f'Total number of batches: {totalBatches}')

        return {'epoch': int(epoch), 'trainLoss': float(trainLoss.result()), 'trainAccuracy': float(trainAccuracy.result()), 'valLoss': float(valLoss.result()), 'valAccuracy': float(valAccuracy.result())}


    def checkResults(self):
        total = 0
        correct = 0

        for batchIdx, (inputs, outputs, _) in enumerate(self.valDS):
            predictions = self(inputs, training=False)

            qOuts = tf.squeeze(outputs[:,0] > 0.5)
            qPreds = tf.squeeze(predictions[:,0] > 0.5)

            falseCount = 0
            trueCount = 0
            for inp, qPred, qOut, pred, out in zip(inputs.numpy(), qPreds.numpy(), qOuts.numpy(), predictions.numpy(), outputs.numpy()):
                total += 1
                if qPred == qOut:
                    correct += 1
                    errorFlag = ''
                else:
                    errorFlag = 'X '

                if qOut:
                    trueCount += 1
                else:
                    falseCount += 1

                logging.info(f'{errorFlag}inp: {inp[0]} {inp[1:3]} {inp[3:6]} {inp[6:8]} out: {out}  pred: {pred}  qOut: {qOut}  qPred: {qPred}')

        logging.info(f'Correct {correct}/{total}')
        logging.info(f'Outputs #allowed={trueCount} #denied={falseCount}')
